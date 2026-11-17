"""One-time export of the phoneme CTC model to ONNX for the Rust server.

Run from the server-rs directory. uv installs the Python packages into a
throwaway environment (these are the versions the export was tested with):

    uv run --no-project --python 3.12 --with torch==2.15.0 --with transformers==5.22.2 --with onnx python tools/export_onnx.py models

Writes model.onnx, vocab.json and config.json into the output directory.
"""

import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2ForCTC

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        return self.model(input_values).logits


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "models")
    model_id = sys.argv[2] if len(sys.argv) > 2 else MODEL_ID
    out.mkdir(parents=True, exist_ok=True)

    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()
    torch.onnx.export(
        Logits(model),
        (torch.randn(1, 16_000),),
        str(out / "model.onnx"),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {1: "samples"}, "logits": {1: "frames"}},
        opset_version=17,
        dynamo=False,
    )
    for name in ("vocab.json", "config.json"):
        shutil.copyfile(hf_hub_download(model_id, name), out / name)
    print("exported to", out.resolve())


if __name__ == "__main__":
    main()
