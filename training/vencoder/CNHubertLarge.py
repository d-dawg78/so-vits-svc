from typing import Any, Optional

import torch
from fairseq import checkpoint_utils  # type: ignore

from training.vencoder.encoder import SpeechEncoder


class CNHubertLarge(SpeechEncoder):
    def __init__(
        self, vec_path: str = "pretrain/chinese-hubert-large-fairseq-ckpt.pt", device: Optional[str] = None
    ) -> None:
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        self.hidden_dim = 1024
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [vec_path],
            suffix="",
        )
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.model = models[0].to(self.dev)
        self.model.eval()  # type: ignore

    def encoder(self, wav: torch.Tensor) -> Any:
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {"source": feats.to(wav.device), "padding_mask": padding_mask.to(wav.device)}
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)  # type: ignore
        return logits[0].transpose(1, 2)
