"""Convert ConvTranspose1d nodes in uh.onnx to ConvTranspose2d (unit H dim)
so the graph compiles on the DirectML EP. Shape-fixing via Unsqueeze/Squeeze
on the tensor streams (weights may come through Mul nodes, so initializers
are left untouched).
"""
import onnx
from onnx import helper, numpy_helper
import numpy as np

SRC = "onnx_artifacts/uhdyn.onnx"
DST = "onnx_artifacts/uhdyn2d.onnx"

model = onnx.load(SRC)
g = model.graph

axes_name = "ct_axes_2d"
axes_init = numpy_helper.from_array(np.array([2], np.int64), axes_name)

new_nodes = []
count = 0
for node in g.node:
    if node.op_type != "ConvTranspose":
        new_nodes.append(node)
        continue
    count += 1
    tag = node.name or f"ct_{count}"
    u_data = f"{tag}_u_data"
    u_w = f"{tag}_u_w"
    s_out = f"{tag}_s"
    new_nodes.append(helper.make_node("Unsqueeze", [node.input[0], axes_name], [u_data]))
    new_nodes.append(helper.make_node("Unsqueeze", [node.input[1], axes_name], [u_w]))
    ct = helper.make_node(
        "ConvTranspose",
        [u_data, u_w] + list(node.input[2:]),
        [s_out],
        name=node.name or tag,
    )
    for a in node.attribute:
        if a.name == "kernel_shape":
            ct.attribute.append(helper.make_attribute("kernel_shape", [1, a.ints[0]]))
        elif a.name == "strides":
            ct.attribute.append(helper.make_attribute("strides", [1, a.ints[0]]))
        elif a.name == "pads":
            ct.attribute.append(helper.make_attribute("pads", [0, a.ints[0], 0, a.ints[1]]))
        elif a.name == "output_padding":
            ct.attribute.append(helper.make_attribute("output_padding", [0, a.ints[0]]))
        elif a.name == "dilations":
            ct.attribute.append(helper.make_attribute("dilations", [1, a.ints[0]]))
        else:
            ct.attribute.append(a)
    new_nodes.append(ct)
    new_nodes.append(helper.make_node("Squeeze", [s_out, axes_name], [node.output[0]]))

g.ClearField("node")
g.node.extend(new_nodes)
g.initializer.append(axes_init)

onnx.checker.check_model(model)
onnx.save(model, DST)
print(f"rewrote {count} ConvTranspose1d -> 2D -> {DST}")
