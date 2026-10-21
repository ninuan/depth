from typing import Optional

from .encoder import Encoder
from .encoder_depthsplat import EncoderDepthSplat, EncoderDepthSplatCfg
from .encoder_mvsplat import EncoderMVSplat, EncoderMVSplatCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_depthsplat import EncoderVisualizerDepthSplat

ENCODERS = {
    "depthsplat": (EncoderDepthSplat, EncoderVisualizerDepthSplat),
    "mvsplat": (EncoderMVSplat, None),
}

EncoderCfg = EncoderDepthSplatCfg | EncoderMVSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
