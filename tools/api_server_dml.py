"""Fish-speech 1.5 TTS API server backed by the DirectML engine.

Drop-in for tools/api_server.py's /v1/tts endpoint: same ServeTTSRequest
msgpack schema, same response shape (audio StreamResponse). Serves on a
different port by default (18081) so it can run alongside the torch server.

Usage:
  python tools/api_server_dml.py --listen 127.0.0.1:18081 \
      --checkpoint checkpoints/fish-speech-1.5 --onnx-dir onnx_artifacts

Test with the stock client:
  python tools/api_client.py --url http://127.0.0.1:18081/v1/tts \
      --text "..." --reference_audio ref.wav --reference_text "..." \
      --output out --no-play
"""
import argparse
import io
import re
import time

import soundfile as sf
import uvicorn
from kui.asgi import (
    Body,
    Depends,
    FactoryClass,
    HTTPException,
    HttpView,
    JSONResponse,
    Kui,
    Routes,
    StreamResponse,
    request,
)
from kui.cors import CORSConfig
from kui.security import bearer_auth
from loguru import logger
from typing_extensions import Annotated

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.utils.schema import ServeTTSRequest
from tools.onnx_tts import OnnxTTS
from tools.server.api_utils import (
    MsgPackRequest,
    buffer_to_async_generator,
    get_content_type,
)
from tools.server.exception_handler import ExceptionHandler

SAMPLE_RATE = 22050

routes = Routes()


@routes.http("/v1/health")
class Health(HttpView):
    @classmethod
    async def get(cls):
        return JSONResponse({"status": "ok"})

    @classmethod
    async def post(cls):
        return JSONResponse({"status": "ok"})


@routes.http.post("/v1/tts")
async def tts(req: Annotated[ServeTTSRequest, Body(exclusive=True)]):
    engine = request.app.state.engine

    if req.streaming and req.format != "wav":
        raise HTTPException(400, content="Streaming only supports WAV format")

    start = time.perf_counter()
    audio = engine.synthesize_request(req)
    logger.info(f"[EXEC] DML TTS time: {(time.perf_counter() - start) * 1000:.0f}ms")

    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format=req.format)

    return StreamResponse(
        iterable=buffer_to_async_generator(buffer.getvalue()),
        headers={
            "Content-Disposition": f"attachment; filename=audio.{req.format}",
        },
        content_type=get_content_type(req.format),
    )


class API(ExceptionHandler):
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--listen", default="127.0.0.1:18081")
        ap.add_argument("--checkpoint", default="checkpoints/fish-speech-1.5")
        ap.add_argument("--onnx-dir", default="onnx_artifacts")
        ap.add_argument("--api-key", default=None)
        self.args = ap.parse_args()

        def api_auth(endpoint):
            async def verify(token: Annotated[str, Depends(bearer_auth)]):
                if token != self.args.api_key:
                    raise HTTPException(401, None, "Invalid token")
                return await endpoint()

            async def passthrough():
                return await endpoint()

            if self.args.api_key is not None:
                return verify
            return passthrough

        self.routes = Routes(
            routes,
            http_middlewares=[api_auth],
        )

        self.app = Kui(
            routes=self.routes,
            exception_handlers={
                HTTPException: self.http_exception_handler,
                Exception: self.other_exception_handler,
            },
            factory_class=FactoryClass(http=MsgPackRequest),
            cors_config=CORSConfig(),
        )
        self.app.on_startup(self.initialize_app)

    async def initialize_app(self, app: Kui):
        t0 = time.perf_counter()
        app.state.engine = OnnxTTS(self.args.checkpoint, self.args.onnx_dir, use_dml=True)
        logger.info(f"DML engine ready in {time.perf_counter() - t0:.1f}s")
        logger.info(f"Startup done, listening server at http://{self.args.listen}")


if __name__ == "__main__":
    api = API()

    match = re.search(r"\[([^\]]+)\]:(\d+)$", api.args.listen)
    if match:
        host, port = match.groups()
    else:
        host, port = api.args.listen.split(":")

    uvicorn.run(
        api.app,
        host=host,
        port=int(port),
        workers=1,
        log_level="info",
    )
