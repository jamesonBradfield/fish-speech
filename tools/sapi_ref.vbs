' Generate speech with the installed Microsoft SAPI 5 voice (prefer Zira = female).
' Usage: cscript //nologo sapi_ref.vbs <out.wav> [text]
Option Explicit
Dim oVoice, oFile, v, outPath, voices, txt, chosen
outPath = WScript.Arguments(0)
If WScript.Arguments.Count > 1 Then
    txt = WScript.Arguments(1)
Else
    txt = "The Earth is being attacked and my systems are being compromised."
End If
Set oVoice = CreateObject("SAPI.SpVoice")
Set oFile = CreateObject("SAPI.SpFileStream")
oFile.Format.Type = 39   ' SAFT22kHz16BitMono
oFile.Open outPath, 3
Set oVoice.AudioOutputStream = oFile
chosen = False
For Each v In oVoice.GetVoices
    If InStr(v.GetDescription, "Zira") > 0 Then
        Set oVoice.Voice = v
        chosen = True
    End If
Next
If Not chosen Then Set oVoice.Voice = oVoice.GetVoices(0)
WScript.Echo "Voice: " & oVoice.Voice.GetDescription
oVoice.Rate = 0
oVoice.Volume = 100
oVoice.Speak txt
oFile.Close
WScript.Echo "Wrote " & outPath
