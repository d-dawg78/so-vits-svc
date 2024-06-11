from typing import Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from training.modules.F0Predictor.crepe import CrepePitchExtractor
from training.modules.F0Predictor.F0Predictor import F0Predictor


class CrepeF0Predictor(F0Predictor):
    def __init__(
        self,
        hop_length: int = 512,
        f0_min: int = 50,
        f0_max: int = 1100,
        device: Optional[str] = None,
        sampling_rate: int = 44100,
        threshold: float = 0.05,
        model: Literal["full", "tiny"] = "full",
    ):
        self.F0Creper = CrepePitchExtractor(
            hop_length=hop_length, f0_min=f0_min, f0_max=f0_max, device=device, threshold=threshold, model=model
        )
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.name = "crepe"

    def compute_f0(self, wav: torch.Tensor, p_len: Optional[int] = None) -> npt.NDArray[np.float32]:
        x = wav.to(self.device)
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        f0, _ = self.F0Creper(x[None, :].float(), self.sampling_rate, pad_to=p_len)
        return f0

    def compute_f0_uv(
        self, wav: torch.Tensor, p_len: Optional[int] = None
    ) -> Tuple[npt.NDArray[np.float32], Optional[npt.NDArray[np.float32]]]:
        x = wav.to(self.device)
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        f0, uv = self.F0Creper(x[None, :].float(), self.sampling_rate, pad_to=p_len)
        return f0, uv
