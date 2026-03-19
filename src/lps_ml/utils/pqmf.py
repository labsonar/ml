"""
This file contains an adapted implementation of the PQMF filter bank
originally developed for the RAVE project.

Original source:
https://github.com/acids-ircam/RAVE/blob/master/rave/pqmf.py

The implementation was modified for integration with the LPS machine learning library.
"""
import math
import torch
import numpy as np

import scipy

# pylint: disable=not-callable
class PQMF(torch.nn.Module):
    """
    Polyphase Quadrature Mirror Filter (PQMF) analysis/synthesis filter bank.

    Splits a full-band audio signal into multiple sub-bands and reconstructs
    the signal from them.

    This implementation is adapted from the PQMF module of RAVE:
    https://github.com/acids-ircam/RAVE/blob/master/rave/pqmf.py

    Parameters
    ----------
    n_band : int
        Number of sub-bands.
    attenuation : int, optional
        Stopband attenuation (dB) used to design the prototype filter.
    """

    def __init__(self, n_band: int, attenuation: int = 100):

        super().__init__()

        h = self._get_prototype(attenuation, n_band)

        h = torch.from_numpy(h).float()

        hk = self._get_qmf_bank(h, n_band)

        hk = self._center_pad_next_pow_2(hk)

        self.register_buffer("hk", hk)

        self.n_band = n_band

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply PQMF analysis.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, time).

        Returns
        -------
        torch.Tensor
            Sub-band representation of shape (batch, n_band * channels, time / n_band).
        """

        x = torch.nn.functional.conv1d(
            x,
            self.hk.unsqueeze(1),
            stride=self.n_band,
            padding=self.hk.shape[-1] // 2,
        )[..., :-1]

        return self._reverse_half(x)

    def reverse(self, x):
        """
        Reconstruct the full-band signal from PQMF sub-bands.

        Parameters
        ----------
        x : torch.Tensor
            Sub-band tensor of shape (batch, n_band * channels, time / n_band).

        Returns
        -------
        torch.Tensor
            Reconstructed signal of shape (batch, channels, time).
        """

        x = self._reverse_half(x)

        hk = self.hk.flip(-1)

        y = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.n_band * x.shape[-1],
        ).to(x)

        y[..., ::self.n_band] = x * self.n_band

        y = torch.nn.functional.conv1d(
            y,
            hk.unsqueeze(0),
            padding=hk.shape[-1] // 2,
        )[..., 1:]

        return y

    @staticmethod
    def _reverse_half(x):
        """Apply alternating sign correction used in PQMF modulation."""
        mask = torch.ones_like(x)
        mask[..., 1::2, ::2] = -1
        return x * mask

    @staticmethod
    def _center_pad_next_pow_2(x):
        """Pad filters so their length is the next power of two."""
        next_2 = 2 ** math.ceil(math.log2(x.shape[-1]))
        pad = next_2 - x.shape[-1]
        return torch.nn.functional.pad(x, (pad // 2, pad // 2 + int(pad % 2)))

    @staticmethod
    def _get_qmf_bank(h, n_band):
        """Generate the modulated PQMF filter bank from the prototype filter."""

        k = torch.arange(n_band).reshape(-1, 1)
        n_samples = h.shape[-1]

        t = torch.arange(-(n_samples // 2), n_samples // 2 + 1)

        p = (-1) ** k * math.pi / 4

        mod = torch.cos((2 * k + 1) * math.pi / (2 * n_band) * t + p)

        hk = 2 * h * mod

        return hk

    @staticmethod
    def _kaiser_filter(wc, atten):
        """Design the prototype FIR filter using a Kaiser window."""

        n, beta = scipy.signal.kaiserord(atten, wc / np.pi)
        n = 2 * (n // 2) + 1

        h = scipy.signal.firwin(
            n,
            wc,
            window=("kaiser", beta),  # type: ignore
            scale=False,
            nyq=np.pi,
        )

        return h

    @staticmethod
    def _loss_wc(wc, atten, m):
        """Evaluate aliasing error for a candidate cutoff frequency."""

        h = PQMF._kaiser_filter(wc, atten)

        g = np.convolve(h, h[::-1], "full")

        g = abs(g[g.shape[-1] // 2::2 * m][1:])

        return np.max(g)

    @staticmethod
    def _get_prototype(atten, m):
        """Compute the prototype filter used to build the PQMF bank."""

        wc = scipy.optimize.fmin(lambda w: PQMF._loss_wc(w, atten, m), 1 / m, disp=0)[0]

        return PQMF._kaiser_filter(wc, atten)
