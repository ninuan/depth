from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_only

LOG_PATH = Path("outputs/local")


class LocalLogger(CSVLogger):
    """Per-run scalar/image logging without requiring a wandb account."""
    def __init__(self, path: Path = LOG_PATH):
        super().__init__(save_dir=str(path), name="", version=None, flush_logs_every_n_steps=10)

    @rank_zero_only
    def log_image(self, key: str, images: list[Any], step: Optional[int] = None, **kwargs):
        assert step is not None
        for index, image in enumerate(images):
            path = Path(self.log_dir) / f"{key}/{index:02}_{step:06}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().permute(1, 2, 0).numpy()
            Image.fromarray(np.asarray(image).astype(np.uint8)).save(path)
