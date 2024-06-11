from typing import Any, Optional

import torch

from training.vencoder.encoder import SpeechEncoder
from training.vencoder.whisper.audio import log_mel_spectrogram, pad_or_trim
from training.vencoder.whisper.model import ModelDimensions, Whisper


class WhisperPPGLarge(SpeechEncoder):
    def __init__(self, vec_path: str = "pretrain/large-v2.pt", device: Optional[str] = None) -> None:
        super().__init__()
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        checkpoint = torch.load(vec_path, map_location=device)
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.hidden_dim = dims  # type: ignore
        self.model = model.to(self.dev)  # type: ignore

    def encoder(self, wav: torch.Tensor) -> Any:
        audio = wav
        audln = audio.shape[0]
        ppgln = audln // 320
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio).to(self.dev)
        with torch.no_grad():
            ppg = self.model.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()  # type: ignore
            ppg = torch.FloatTensor(ppg[:ppgln,]).to(self.dev)
            return ppg[None, :, :].transpose(1, 2)
