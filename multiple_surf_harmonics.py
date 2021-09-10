"""
Calculate the harmonics of various quantities on multiple flux surface
"""
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
from covariant_b_components import read_geqdsk
from pleque.tests.utils import get_test_equilibria_filenames
from surf_harmonics import SurfaceHarmonics, SurfaceData, _check_equal_max_harmonics
from more_itertools import all_equal


class CollectionOfSurfaceHarmonics:

    def __init__(self, s, surf_harmonics_list):

        _check_equal_max_harmonics(surf_harmonics_list)
        self.s = s
        self._surf_harmonics = surf_harmonics_list


    @staticmethod
    def _calc_surface_harmonics(eq, psi_n, max_harmonic):
        surf = SurfaceData(eq._flux_surface(psi_n=psi_n)[0])

        return  SurfaceHarmonics.from_eq(eq, surf, max_harmonic)

    @staticmethod
    def _concat_ampls(amplitudes):
        return np.column_stack(amplitudes)

    def _get_cos_ampls(self, quantity):

        cos_ampls = [getattr(sh, quantity).Cn for sh in self]

        return self._concat_ampls(cos_ampls)

    def _get_sin_ampls(self, quantity):

        sin_ampls = [getattr(sh, quantity).Sn for sh in self]

        return self._concat_ampls(sin_ampls)

    @classmethod
    def from_eq(cls, s_in, eq, max_harmonic):
        s = np.array(s_in).ravel()

        surf_harmonics=[cls._calc_surface_harmonics(eq,
                                                    si,
                                                    max_harmonic)
                        for si in s]

        return cls(s, surf_harmonics)

    def __iter__(self):
        return iter(self._surf_harmonics)

    @property
    def max_harmonic(self):
        return self._surf_harmonics[0].max_harmonic

    @property
    def mode_numbers(self):
        """
        The mode numbers of the contained Fourier series
        """
        return np.arange(self.max_harmonic+1)

    @property
    def bmnc(self):
        return self._get_cos_ampls('B_abs')

    @property
    def bmns(self):
        return self._get_sin_ampls('B_abs')

    @property
    def bsubumnc(self):
        return self._get_cos_ampls('B_theta')

    @property
    def bsubumns(self):
        return self._get_sin_ampls('B_theta')

    @property
    def bsubvmnc(self):
        return self._get_cos_ampls('B_zeta')

    @property
    def bsubvmns(self):
        return self._get_sin_ampls('B_zeta')

    @property
    def lmnc(self):
        return self._get_cos_ampls('lambda_shift')

    @property
    def lmns(self):
        return self._get_sin_ampls('lambda_shift')

    @property
    def rmnc(self):
        return self._get_cos_ampls('R')

    @property
    def rmns(self):
        return self._get_sin_ampls('R')

    @property
    def zmnc(self):
        return self._get_cos_ampls('Z')

    @property
    def zmns(self):
        return self._get_sin_ampls('Z')

if __name__ == "__main__":
    gfiles = get_test_equilibria_filenames()

    test_case = 5
    FILEPATH = gfiles[test_case]

    # Create an instance of the `Equilibrium` class
    eq = read_geqdsk(FILEPATH)

    s_in = np.linspace(0.1, 0.8, num=3)

    col = CollectionOfSurfaceHarmonics.from_eq(s_in, eq, 15)
