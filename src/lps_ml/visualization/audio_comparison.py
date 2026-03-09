import enum
import numpy as np
from scipy.signal import stft
from scipy.linalg import sqrtm


class ReconstructionMetric(enum.Enum):
    MSE = enum.auto()
    MAE = enum.auto()
    PSNR = enum.auto()
    SI_SDR = enum.auto()
    MULTI_SCALE_SPECTRAL = enum.auto()

    def apply(self,
              x: np.ndarray,
              y: np.ndarray,
              fs: int = 16000) -> float:
        """
        Args:
            x: sinal original
            x_hat: sinal reconstruído
            fs: frequência de amostragem (necessária para métricas espectrais)

        Returns:
            float: valor da métrica
        """

        assert x.shape == y.shape, "Shapes devem ser iguais"

        x = np.asarray(x).astype(np.float64)
        y = np.asarray(y).astype(np.float64)

        if self == ReconstructionMetric.MSE:
            return np.mean((x - y) ** 2)

        elif self == ReconstructionMetric.MAE:
            return np.mean(np.abs(x - y))

        elif self == ReconstructionMetric.PSNR:
            mse = np.mean((x - y) ** 2)
            if mse == 0:
                return np.inf
            max_val = np.max(np.abs(x))
            return 20 * np.log10(max_val / np.sqrt(mse))

        elif self == ReconstructionMetric.SI_SDR:
            return self._si_sdr(x, y)

        elif self == ReconstructionMetric.MULTI_SCALE_SPECTRAL:
            return self._multi_scale_spectral(x, y, fs)

        else:
            raise NotImplementedError


    @staticmethod
    def _si_sdr(x, x_hat):
        """
        Scale-Invariant SDR
        """
        x = x.reshape(-1)
        x_hat = x_hat.reshape(-1)

        alpha = np.dot(x_hat, x) / np.dot(x, x)
        s_target = alpha * x
        e_noise = x_hat - s_target

        return 10 * np.log10(
            np.sum(s_target ** 2) / np.sum(e_noise ** 2)
        )

    @staticmethod
    def _multi_scale_spectral(x, x_hat, fs):
        """
        Distância espectral multi-escala usando STFT
        """
        scales = [256, 512, 1024]
        total_distance = 0.0

        for n_fft in scales:
            _, _, Zxx = stft(x, fs=fs, nperseg=n_fft)
            _, _, Zyy = stft(x_hat, fs=fs, nperseg=n_fft)

            mag_x = np.abs(Zxx)
            mag_y = np.abs(Zyy)

            total_distance += np.mean(np.abs(mag_x - mag_y))

        return total_distance / len(scales)