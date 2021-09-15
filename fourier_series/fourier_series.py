import numpy as np
from fourier_series.calculators import SimpsonFourierCalculator as FC
from numba import njit, guvectorize

class FourierSeries():
    calculator = FC()

    def __init__(self, Cn, Sn):
        Cn = np.array(Cn).ravel()
        Sn = np.array(Sn).ravel()

        if not Cn.size == Sn.size:
            msg = "The sizes of Cn and Sn must be equal."
            raise ValueError(msg)

        self.Cn = Cn
        self.Sn = Sn

    @property
    def max_harmonic(self):
        return self.Cn.size-1

    def __call__(self, theta):
        return _fast_evaluate_fourier_series(self.Cn, self.Sn, theta)

    @classmethod
    def from_data(cls, theta, y, n):
        return cls(*cls.calculator(theta, y, n))


def _evaluate_fourier_series(Cn, Sn, theta):
    """
    Evaluate the Fourier Series represented by `Cn`, `Sn` at `theta`.

    Parameters
    ----------
    Cn, Sn: array, shape(M)
        The coefficients of the Fourier series.

    theta: scalar or array, shape(N)
        The angles.

    Returns
    -------
    out: scalar or array, shape(N)
        The values of the Fourier series.
    """

    return sum((Cn[n]*np.cos(n*theta) + Sn[n]*np.sin(n*theta)
                for n in range(Cn.size)))


@njit()
def _fast_sum_fourier_series(Cn, Sn, theta, out):
    for n in range(Cn.size):
        out += Cn[n]*np.cos(n*theta) + Sn[n]*np.sin(n*theta)


_fast_scalar_evaluate_fourier_series = njit()(_evaluate_fourier_series)


def _fast_evaluate_fourier_series(Cn, Sn, theta):
    """
    Evaluate the Fourier Series represented by `Cn`, `Sn` at `theta`.

    This is equivalent to `_evaluate_fourier_series`, only it uses
    a numba compiled implementation for better performance

    Parameters
    ----------
    Cn, Sn: array, shape(M)
        The coefficients of the Fourier series.

    theta: scalar or array, shape(N)
        The angles.

    Returns
    -------
    out: scalar or array, shape(N)
        The values of the Fourier series.
    """
    if np.isscalar(theta):
        out = np.array([0], dtype=float)
        _fast_sum_fourier_series(Cn, Sn, theta, out)
        return out[0]

    out = np.zeros_like(theta, dtype=float)

    _fast_sum_fourier_series(Cn, Sn, theta, out)
    return out


