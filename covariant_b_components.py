"""
Calculate the covariant components of B in magnetic coordinates using PLEQUE
"""
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
import pleque
import pleque.io.readers as readers
from pleque.tests.utils import get_test_equilibria_filenames, load_testing_equilibrium
from pleque.core.cocos import cocos_coefs
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import CubicSpline


class Equilibrium():
    """
    A composed class to add functionality to pleque.Equilibrium.
    Inheritance is not used, because this approach makes it easier to use
    pleque's constructors.
    For more see https://stackoverflow.com/a/69028698/6060982
    """
    def __init__(self,
                 *args,
                 consume_pleque_equilibrium=None,
                 **kwargs):

        if consume_pleque_equilibrium is None:
            self.eq = pleque.Equilibrium(*args, **kwargs)

        else:
            self.eq = consume_pleque_equilibrium

    def __getattr__(self, name):
        return getattr(self.eq, name)

    def get_cc_coef(self):
        cocos_dict = cocos_coefs(self.cocos)
        cc_coef = - cocos_dict['sigma_pol'] * cocos_dict['sigma_cyl']
        return cc_coef

    def grad_theta(self, R, Z):
        cc_coef = self.get_cc_coef()
        r0 = self.magnetic_axis.R
        z0 = self.magnetic_axis.Z

        r = R - r0
        z = Z - z0

        theta_R = - cc_coef * z/(r**2 + z**2)
        theta_Z = cc_coef * r/(r**2 + z**2)

        return theta_R, theta_Z

    def grad_psi(self, R, Z):
        psi_R = self._spl_psi(R, Z, dx=1, grid=False)
        psi_Z = self._spl_psi(R, Z, dy=1, grid=False)

        return psi_R, psi_Z

    def get_B_pol_covariant_components(self, R, Z):
        """
        Calculate the covariant components of the poloidal magnetic field in
        flux coordinates.

        Returns
        -------
        B_psi, B_theta: array,
            The poloidal magnetic field components.
        """

        B_R = self.B_R(R, Z, grid=False)
        B_Z = self.B_Z(R, Z, grid=False)

        psi_R, psi_Z = self.grad_psi(R, Z)
        theta_R, theta_Z = self.grad_theta(R, Z)

        D = psi_Z * theta_R - psi_R * theta_Z

        B_theta = (B_R * psi_Z - B_Z * psi_R)/D

        B_psi = (B_Z * theta_R - B_R * theta_Z)/D

        return B_psi, B_theta


def read_geqdsk(filename, cocos=3):
    return Equilibrium(
        consume_pleque_equilibrium=readers.read_geqdsk(filename,
                                                       cocos)
    )


if __name__ == "__main__":

    gfiles = get_test_equilibria_filenames()

    test_case = 5
    FILEPATH = gfiles[test_case]

    # Create an instance of the `Equilibrium` class
    eq = read_geqdsk(FILEPATH)


    surf = eq._flux_surface(psi_n=0.7)[0]


    R = surf.R
    Z = surf.Z

    B_psi, B_theta = eq.get_B_pol_covariant_components(R, Z)


    theta_R, theta_Z = eq.grad_theta(R, Z)

    B_R = eq.B_R(R, Z, grid=False)
    B_Z = eq.B_Z(R, Z, grid=False)
    psi_R, psi_Z = eq.grad_psi(R, Z)
    D = psi_Z * theta_R - psi_R * theta_Z
    B_R_again = B_theta * theta_R + B_psi * psi_R
    B_Z_again = B_theta * theta_Z + B_psi * psi_Z

    assert np.allclose(0, B_R - B_R_again, rtol=1e-12, atol=1e-12)
    assert np.allclose(0, B_Z - B_Z_again, rtol=1e-12, atol=1e-12)

    fig, ax = plt.subplots()
    ax.quiver(R[::100], Z[::100], psi_R[::100], psi_Z[::100])
    ax.quiver(R[::100], Z[::100], B_R[::100], B_Z[::100])
    ax.plot(surf.R, surf.Z)
    ax.set_aspect("equal")

    fig, ax = plt.subplots()
    ax.plot(np.abs(B_theta) * np.hypot(theta_R, theta_Z), label='B_theta measure')
    ax.plot(np.abs(B_psi) * np.hypot(psi_R, psi_Z), label='B_psi measure')
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(theta_R, label='theta_R')
    ax.plot(psi_R, label='psi_R')
    ax.legend()
    plt.show()

