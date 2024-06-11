from typing import Any, Optional

import torch

from training.vencoder.encoder import SpeechEncoder
from training.vencoder.wavlm.WavLM import WavLM, WavLMConfig


class WavLMBasePlus(SpeechEncoder):
    def __init__(self, vec_path: str = "pretrain/WavLM-Base+.pt", device: Optional[str] = None) -> None:
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        checkpoint = torch.load(vec_path)
        self.cfg = WavLMConfig(checkpoint["cfg"])  # type: ignore
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.hidden_dim = self.cfg.encoder_embed_dim
        self.model = WavLM(self.cfg)  # type: ignore
        self.model.load_state_dict(checkpoint["model"])  # type: ignore
        self.model.to(self.dev).eval()  # type: ignore

    def encoder(self, wav: torch.Tensor) -> Any:
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        if self.cfg.normalize:
            feats = torch.nn.functional.layer_norm(feats, feats.shape)
        with torch.inference_mode():
            units = self.model.extract_features(feats[None, :])[0]  # type: ignore
            return units.transpose(1, 2)
