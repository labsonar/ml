import typing

import numpy as np

import lps_utils.quantities as lps_qty
import lps_sp.acoustical.analysis as lps_analysis
import lps_ml.core.processor as ml_core


class SpectralProcessor(ml_core.AudioPipeline):
    """
    Pipeline que aplica uma análise espectral e retorna o espectrograma
    como saída (tratado como novo 'sinal').
    """

    def __init__(
        self,
        analysis: lps_analysis.SpectralAnalysis,
        params: lps_analysis.Parameters | None
    ):
        super().__init__()

        self.analysis = analysis
        self.params = params or lps_analysis.Parameters()

    def process(
        self,
        fs: lps_qty.Frequency,
        data: np.ndarray
    ) -> typing.Tuple[lps_qty.Frequency, np.ndarray]:

        fs_hz = fs.get_hz() if isinstance(fs, lps_qty.Frequency) else fs

        power, _, _ = self.analysis.apply(
            data=data,
            fs=fs_hz,
            params=self.params
        )

        return fs, power
