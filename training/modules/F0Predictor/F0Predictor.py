from typing import Any, Optional

import torch


class F0Predictor(object):
    def compute_f0(self, wav: torch.Tensor, p_len: Optional[int]) -> Any:
        """
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length]
        """
        pass

    def compute_f0_uv(self, wav: torch.Tensor, p_len: Optional[int]) -> Any:
        """
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length],uv:[signal_length//hop_length]
        """
        pass
