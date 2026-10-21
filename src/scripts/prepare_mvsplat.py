"""Download/check the official MVSplat RE10K weights; no GPU is needed."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.misc.mvsplat_checkpoint import OFFICIAL_FILE_ID, OFFICIAL_SHA256, load_mvsplat_weights
from src.model.encoder.encoder_mvsplat import EncoderMVSplat, EncoderMVSplatCfg
from src.model.encoder.mvsplat.common.gaussian_adapter import GaussianAdapterCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained/mvsplat-re10k.ckpt"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    path = args.checkpoint
    if not path.is_file() and args.download:
        import gdown  # Optional; only needed to fetch the official checkpoint.
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        downloaded = gdown.download(id=OFFICIAL_FILE_ID, output=str(temporary),
                                    use_cookies=False, resume=True)
        if not downloaded:
            raise RuntimeError("Download failed; retry or upload the official re10k.ckpt")
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != OFFICIAL_SHA256:
            raise ValueError(f"Downloaded checksum mismatch: {temporary}; remove it and retry")
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OFFICIAL_SHA256:
        raise ValueError("Expected the official default RE10K checkpoint, not an ablation or fine-tuned model")
    # The official Lightning checkpoint is trusted only after its hash is verified.
    weights = torch.load(path, map_location="cpu", weights_only=False)
    encoder = EncoderMVSplat(EncoderMVSplatCfg("mvsplat", GaussianAdapterCfg(.5, 15., 4)))
    load_mvsplat_weights(encoder, weights)
    report = dict(checkpoint=str(path), sha256=digest, encoder_keys=len(encoder.state_dict()),
                  parameters=sum(p.numel() for p in encoder.parameters()),
                  source="https://github.com/donydchen/mvsplat", file_id=OFFICIAL_FILE_ID)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("PASS: official checksum and strict encoder weights")


if __name__ == "__main__":
    main()
