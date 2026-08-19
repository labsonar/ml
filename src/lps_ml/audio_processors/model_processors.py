import os
import typing
import torch
import numpy as np

import lps_ml.utils.device as lps_device
import lps_utils.quantities as lps_qty
import lps_ml.core as ml_core
import lps_ml.model.audio_vae as lps_audio_vae
import lps_ml.model.cnn as lps_cnn


class VAEEncoder(ml_core.AudioPipeline):
    """ Audio Pipeline that projects the signal into latent space using a VAE (e.g., RAVE). """

    def __init__(
        self,
        model_path: str,
        device: str | None= None
    ):
        super().__init__()
        self.device = device or lps_device.get_available_device()
        self.model_path = model_path

        ext = os.path.splitext(model_path)[1]
        if ext == ".ts":
            self.model = torch.jit.load(model_path).to(self.device)
        elif ext == ".ckpt":
            self.model = lps_audio_vae.CONV_VAE.load_from_checkpoint(model_path)
        else:
            raise NotImplementedError(f"VAEEncoder not ready to load an {ext} file")

        self.model.eval()

        if not hasattr(self.model, "decode"):
            print("[WARNING] Model has no decode() method")

    def decode(self, z: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Decode latent representation back to waveform.
        """
        input_is_numpy = isinstance(z, np.ndarray)

        if input_is_numpy:
            z = torch.from_numpy(z)

        z = z.to(self.device)

        if z.ndim == 2:
            z = z.unsqueeze(0)

        with torch.inference_mode():
            x = self.model.decode(z)

        if input_is_numpy:
            return x.detach().cpu().numpy()

        if x.ndim == 3 and x.shape[1] == 1:
            x = torch.squeeze(x, dim=1)
        if x.ndim == 2 and x.shape[0] == 1:
            x = torch.squeeze(x, dim=0)

        if input_is_numpy:
            return x.detach().cpu().numpy()

        return x

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

    def _get_params(self):
        return {
            "Pipeline": "VAEEncoder",
            "model_path": self.model_path,
        }

class CNN2DPipeline(ml_core.SamplePipeline):
    """
    SamplePipeline that extracts and concatenates embeddings
    from one or more CNN2D models.
    """

    def __init__(self,
                 model_paths: typing.Sequence[str],
                 flatten: bool = False,
                 device: str | None = None):
        super().__init__()

        if not model_paths:
            raise ValueError("At least one CNN model must be provided.")

        self.model_paths = list(model_paths)
        self.flatten = flatten
        self.device = device or lps_device.get_available_device()

        self.models = torch.nn.ModuleList()

        for model_path in self.model_paths:

            model = lps_cnn.CNN2D.load_from_checkpoint(model_path)
            model.to(self.device)
            model.eval()

            for parameter in model.parameters():
                parameter.requires_grad = False

            self.models.append(model)

    def process(self, fs: lps_qty.Frequency, data: np.ndarray) -> \
            typing.Tuple[lps_qty.Frequency, np.ndarray]:

        if data.ndim == 2:
            # [F, T] -> [B=1, C=1, F, T]
            x = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

        elif data.ndim == 3:
            # [C, F, T] -> [B=1, C, F, T]
            x = torch.from_numpy(data).float().unsqueeze(0)

        else:
            raise ValueError(
                f"CNN2DPipeline expects input shape [F, T] or [C, F, T], got {tuple(data.shape)}"
            )

        x = x.to(self.device)

        embeddings = []

        with torch.inference_mode():

            for model in self.models:
                # [1, C, F, T] -> [1, C, H, W]
                z = model.to_feature_space(x)

                embeddings.append(z)

        # [1, C1, H, W], [1, C2, H, W], ...
        # -> [1, C1+C2+..., H, W]
        embedding = torch.cat(embeddings, dim=1)

        # [1, C, H, W] -> [C, H, W]
        embedding = embedding.squeeze(0)

        if self.flatten:
            # [C, H, W] -> [C*H*W]
            embedding = torch.flatten(embedding)

        embedding = embedding.cpu().numpy()

        return fs, embedding

    def _get_params(self):
        return {
            "Pipeline": "CNN2DPipeline",
            "model_paths": self.model_paths,
            "flatten": self.flatten,
        }
