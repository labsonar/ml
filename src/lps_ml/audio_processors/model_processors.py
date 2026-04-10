import typing
import torch
import numpy as np

import lps_ml.utils.device as lps_device
import lps_utils.quantities as lps_qty
import lps_ml.core as ml_core


class VAEEncoder(ml_core.AudioPipeline):
    """ Audio Pipeline that projects the signal into latent space using a VAE (e.g., RAVE). """

    def __init__(
        self,
        model_path: str,
        device: str | None= None
    ):
        super().__init__()
        self.device = device or lps_device.get_available_device()
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()

        if not hasattr(self.model, "encode"):
            raise RuntimeError(
                "Modelo não possui método encode(). "
                "Use um modelo RAVE exportado corretamente."
            )


    def process(
        self,
        fs: lps_qty.Frequency,
        data: np.ndarray
    ) -> typing.Tuple[lps_qty.Frequency, np.ndarray]:

        x = torch.from_numpy(data).to(self.device)

        print("##########")
        print("x: ", x.shape)

        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)

        elif x.ndim == 2:
            x = x.unsqueeze(1)

        print("x: ", x.shape)

        with torch.inference_mode():
            z = self.model.encode(x)

        print("z: ", z.shape)

        z = z.detach().cpu().numpy()
        z = np.squeeze(z, axis=0)
        return fs, z
