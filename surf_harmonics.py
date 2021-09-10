"""
Calculate the harmonics of various quantities on a flux surface
"""
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
from covariant_b_components import read_geqdsk
from pleque.tests.utils import get_test_equilibria_filenames
from fourier_series import SimpsonFourierCalculator as FC




class SurfaceData():

    def __init__(self, surf, atol=1e-4):
        self.surf = surf
        self.atol = atol
        self._theta = np.mod(surf.theta, 2*np.pi)
        self.asort = np.argsort(self._theta)
        self.mask = ~ self._duplicated_value(self._theta[self.asort])


    def _duplicated_value(self, x):
        """
        returns an boolean array with elements being true when the value of the
        corresponding element in the x vector is equal to that of the
        preceding element with tolerance atol.

        [1, 2, 2, 4] -> [False, False, True, False]
        """
        out = (np.diff(x) <= self.atol)
        return np.insert(out, 0, False)

    @property
    def theta(self):
        return self._theta[self.asort][self.mask]

    @property
    def R(self):
        return self.surf.R[self.asort][self.mask]

    @property
    def Z(self):
        return self.surf.Z[self.asort][self.mask]

    @property
    def straight_fieldline_theta(self):
        return self.surf.straight_fieldline_theta[self.asort][self.mask]

    @property
    def lambda_shift(self):
        """
        The straight field line theta shift.
        """
        return self.straight_fieldline_theta - self.theta


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

    def max_harmonic(self):
        return self.Cn.size

    def __call__(self, theta):
        out = 0
        for n in range(self.Cn.size):
            out += self.Cn[n]*np.cos(n*theta) + self.Sn[n]*np.sin(n*theta)

        return out

    @classmethod
    def from_data(cls, theta, y, n):
        return cls(*cls.calculator(theta, y, n))


class SurfaceHarmonics():
    """
    A collection of fourier series for magnetic quantities on a flux surface.
    """

    def __init__(self, lambda_shift, R, Z, B_psi, B_theta, B_abs):
        self.lambda_shift = lambda_shift
        self.R = R
        self.Z = Z
        self.B_psi = B_psi
        self.B_theta = B_theta
        self.B_abs = B_abs


    @classmethod
    def from_eq(cls, eq, surf, max_harmonic):
        """
        Construct on a given flux surface.

        Parameters:
        -----------
        eq:
            The equilibrium
        surf:
            The flux surface

        max_harmonic: int, positive
            The maximum harmonic number in the series

        """

        theta=surf.theta
        lambda_series = FourierSeries.from_data(theta,
                                                surf.lambda_shift,
                                                max_harmonic)

        R_series = FourierSeries.from_data(theta,
                                           surf.R,
                                           max_harmonic)

        Z_series = FourierSeries.from_data(theta,
                                           surf.Z,
                                           max_harmonic)

        B_psi, B_theta = eq.get_B_pol_covariant_components(
            surf.R,
            surf.Z,
        )

        B_psi_series = FourierSeries.from_data(theta,
                                           B_psi,
                                           max_harmonic)

        B_theta_series = FourierSeries.from_data(theta,
                                           B_theta,
                                           max_harmonic)

        B_abs = eq.B_abs(surf.R, surf.Z, grid=False)

        B_abs_series = FourierSeries.from_data(theta,
                                            B_abs,
                                               max_harmonic)

        return cls(
            lambda_shift=lambda_series,
            R=R_series,
            Z=Z_series,
            B_psi=B_psi_series,
            B_theta=B_theta_series,
            B_abs=B_abs_series,
        )


if __name__ == "__main__":
    gfiles = get_test_equilibria_filenames()

    test_case = 5
    FILEPATH = gfiles[test_case]

    # Create an instance of the `Equilibrium` class
    eq = read_geqdsk(FILEPATH)

    surf = SurfaceData(eq._flux_surface(psi_n=0.7)[0])

    theta=surf.theta

    harmonics = SurfaceHarmonics.from_eq(eq, surf, 15)

    lambda_reconstructed = harmonics.lambda_shift(theta)

    fig, ax = plt.subplots()
    ax.plot(theta, surf.lambda_shift, label='initial')
    ax.plot(theta, lambda_reconstructed, label='reconstructed')
    ax.legend()

    R_reconstructed = harmonics.R(theta)
    Z_reconstructed = harmonics.Z(theta)

    fig, ax = plt.subplots()
    ax.plot(surf.R, surf.Z, label='initial')
    ax.plot(R_reconstructed, Z_reconstructed, label='reconstructed')
    ax.legend()
    ax.set_aspect('equal')

    fig, ax = plt.subplots()
    B_psi, B_theta = eq.get_B_pol_covariant_components(surf.R, surf.Z)
    B_psi_reconstructed = harmonics.B_psi(theta)
    B_theta_reconstructed = harmonics.B_theta(theta)

    ax.plot(theta, B_theta, label='initial')
    ax.plot(theta, B_theta_reconstructed, label='reconstructed')
    ax.legend()

    fig, ax = plt.subplots()
    B_abs = eq.B_abs(surf.R, surf.Z, grid=False)
    B_abs_reconstructed = harmonics.B_abs(theta)

    ax.plot(theta, B_abs, label='initial')
    ax.plot(theta, B_abs_reconstructed, label='reconstructed')
    ax.legend()

    plt.show()

