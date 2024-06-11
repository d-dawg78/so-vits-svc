from typing import Any, Optional

import torch


class SpeechEncoder(object):
    def __init__(self, vec_path: str = "pretrain/checkpoint_best_legacy_500.pt", device: Optional[str] = None) -> None:
        self.model = None  # This is Model
        self.hidden_dim = 768
        pass

    def encoder(self, wav: torch.Tensor) -> Any:
        """
        input: wav:[signal_length]
        output: embedding:[batchsize,hidden_dim,wav_frame]
        """
        pass
