import os
import typing
import torch
import numpy as np

import lps_ml.utils.device as lps_device
import lps_utils.quantities as lps_qty
import lps_ml.core as ml_core
import lps_ml.model.audio_vae as lps_audio_vae


class VAEEncoder(ml_core.AudioPipeline):
    """ Audio Pipeline that projects the signal into latent space using a VAE (e.g., RAVE). """

    def __init__(
        self,
        model_path: str,
        device: str | None= None
    ):
        super().__init__()
        self.device = device or lps_device.get_available_device()

        ext = os.path.splitext(model_path)[1]
        if ext == ".ts":
            self.model = torch.jit.load(model_path).to(self.device)
        elif ext == ".ckpt":
            print("ckpt load: ", model_path)
            self.model = lps_audio_vae.CONV_VAE.load_from_checkpoint(model_path)
            print("ckpt loaded")
        else:
            raise NotImplementedError(f"VAEEncoder not ready to load an {ext} file")

        self.model.eval()

        if not hasattr(self.model, "decode"):
            print("[WARNING] Model has no decode() method")

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent representation back to waveform.
        """

        z = torch.from_numpy(z).to(self.device)

        if z.ndim == 2:
            z = z.unsqueeze(0)

        with torch.inference_mode():
            x = self.model.decode(z)

        x = x.detach().cpu().numpy()
        return np.squeeze(x, axis=0)

    def process(
        self,
        fs: lps_qty.Frequency,
        data: np.ndarray
    ) -> typing.Tuple[lps_qty.Frequency, np.ndarray]:

        x = torch.from_numpy(data).to(self.device)

        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)

        elif x.ndim == 2:
            x = x.unsqueeze(1)

        with torch.inference_mode():
            z = self.model.encode(x)

        z = z.detach().cpu().numpy()
        z = np.squeeze(z, axis=0)
        return fs, z
