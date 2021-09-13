"""
Calculate the harmonics of various quantities on a flux surface
"""
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
from covariant_b_components import read_geqdsk
from pleque.tests.utils import get_test_equilibria_filenames
from fourier_series import SimpsonFourierCalculator as FC
from more_itertools import all_equal




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

    @property
    def max_harmonic(self):
        return self.Cn.size-1

    def __call__(self, theta):
        out = 0
        for n in range(self.Cn.size):
            out += self.Cn[n]*np.cos(n*theta) + self.Sn[n]*np.sin(n*theta)

        return out

    @classmethod
    def from_data(cls, theta, y, n):
        return cls(*cls.calculator(theta, y, n))


def _check_equal_max_harmonics(iter_harmonics):
    """
    Raises ValueError if all the elements in iter_harmonics do not have the
    same max_harmonic
    """
    if not all_equal(h.max_harmonic for h in iter_harmonics):
        msg = 'All series must have the same max_harmonic.'
        raise ValueError(msg)


class SurfaceHarmonics():
    """
    A collection of fourier series for magnetic quantities on a flux surface.
    """

    def __init__(self, lambda_shift, R, Z, B_psi, B_theta, B_zeta, B_abs):

        inputs = (lambda_shift, R, Z, B_psi, B_theta, B_zeta, B_abs)

        _check_equal_max_harmonics(inputs)

        self.lambda_shift = lambda_shift
        self.R = R
        self.Z = Z
        self.B_psi = B_psi
        self.B_theta = B_theta
        self.B_zeta = B_zeta
        self.B_abs = B_abs

    @property
    def max_harmonic(self):
        return self.R.max_harmonic


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

        B_zeta = eq.F(surf.R, surf.Z, grid=False)

        B_zeta_series = FourierSeries.from_data(theta,
                                                B_zeta,
                                                max_harmonic)
        return cls(
            lambda_shift=lambda_series,
            R=R_series,
            Z=Z_series,
            B_psi=B_psi_series,
            B_theta=B_theta_series,
            B_zeta=B_zeta_series,
            B_abs=B_abs_series,
        )


if __name__ == "__main__":
    gfiles = get_test_equilibria_filenames()

    test_case = 5
    FILEPATH = gfiles[test_case]

    # Create an instance of the `Equilibrium` class
    eq = read_geqdsk(FILEPATH)

    eq.plot_overview()

    surf = SurfaceData(eq._flux_surface(psi_n=0.7)[0], atol=1e-5)

    theta=surf.theta

    harmonics = SurfaceHarmonics.from_eq(eq, surf, 15)

    lambda_reconstructed = harmonics.lambda_shift(theta)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,12))
    ax = ax.ravel()

    ax[0].plot(theta, surf.lambda_shift, label='initial')
    ax[0].plot(theta, lambda_reconstructed, label='reconstructed')
    ax[0].set_title('lambda shift')
    ax[0].set_xlabel(R"$\theta$")
    ax[0].set_ylabel(R"$\lambda$", rotation=0)
    ax[0].legend()

    R_reconstructed = harmonics.R(theta)
    Z_reconstructed = harmonics.Z(theta)


    ax[1].plot(surf.R, surf.Z, label='initial')
    ax[1].plot(R_reconstructed, Z_reconstructed, label='reconstructed')
    ax[1].set_title("flux surface")
    ax[1].set_xlabel("R")
    ax[1].set_ylabel("Z", rotation=0)
    ax[1].legend()
    ax[1].set_aspect('equal')

    B_psi, B_theta = eq.get_B_pol_covariant_components(surf.R, surf.Z)
    B_psi_reconstructed = harmonics.B_psi(theta)
    B_theta_reconstructed = harmonics.B_theta(theta)

    ax[2].plot(theta, B_theta, label='initial')
    ax[2].plot(theta, B_theta_reconstructed, label='reconstructed')
    ax[2].set_title("B theta covariant")
    ax[2].set_xlabel(r"$\theta$")
    ax[2].set_ylabel(r"$B_\theta$", rotation=0)
    ax[2].legend()

    B_abs = eq.B_abs(surf.R, surf.Z, grid=False)
    B_abs_reconstructed = harmonics.B_abs(theta)

    ax[3].plot(theta, B_abs, label='initial')
    ax[3].plot(theta, B_abs_reconstructed, label='reconstructed')
    ax[3].set_title("magnitude of B")
    ax[3].set_xlabel(r"$\theta$")
    ax[3].set_ylabel(r"$\left|B\right|$", rotation=0)
    ax[3].legend()

    plt.tight_layout()
    plt.show()

