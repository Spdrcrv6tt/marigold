#!/bin/bash
# Fetch the commercially clean TTS models (licenses checked in docs/licenses.md).
# Piper voices -> data/tts/piper/models ; Kokoro-82M -> HF cache data/hf (HF_HOME).
set -eu
R=/d/marigold; P=$R/data/tts/piper; mkdir -p $P/models $P/cards
VOICES="en_US-libritts-high en_US-ljspeech-high en_US-hifi_tts_92-medium en_GB-cori-high en_GB-alba-medium en_GB-jenny_dioco-medium en_US-norman-medium en_US-kristin-medium en_US-john-medium en_US-joe-medium en_US-mike-medium"
for v in $VOICES; do
  loc=${v%%-*}; rest=${v#*-}; name=${rest%-*}; q=${rest##*-}
  base="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/$loc/$name/$q"
  for f in "$v.onnx" "$v.onnx.json"; do [ -s "$P/models/$f" ] || curl -sSfL --retry 3 -o "$P/models/$f" "$base/$f"; done
  [ -s "$P/cards/$v.md" ] || curl -sSfL -o "$P/cards/$v.md" "$base/MODEL_CARD"
done
HF_HOME=$R/data/hf $R/.venv/Scripts/python.exe -I -c "
from huggingface_hub import snapshot_download as s
p=s('hexgrad/Kokoro-82M', allow_patterns=['config.json','kokoro-v1_0.pth','README.md','VOICES.md','voices/a*','voices/b*'])
print(p)"
