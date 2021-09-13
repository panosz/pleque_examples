"""
Read a booz_xform equilibrium from geqdsk data.
"""
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
from covariant_b_components import read_geqdsk
from pleque.tests.utils import get_test_equilibria_filenames
import booz_xform
from multiple_surf_harmonics import CollectionOfSurfaceHarmonics

class Booz_xform(booz_xform.Booz_xform):


    def init_from_geqdsk(self, FILEPATH, psi_n_in):
        eq = read_geqdsk(FILEPATH)

        psi_lcfs = eq.lcfs.psi[0]

        iota = 1/eq.q(psi_n_in)

        col = CollectionOfSurfaceHarmonics.from_eq(psi_n_in, eq, 7)

        self.asym = True
        self.bmnc = col.bmnc
        self.bmns = col.bmns
        self.bsubumnc = col.bsubumnc
        self.bsubumns = col.bsubumns
        self.bsubvmnc = col.bsubvmnc
        self.bsubvmns = col.bsubvmns
        self.compute_surfs = np.arange(psi_n_in.size)
        self.iota = iota
        self.lmnc = col.lmnc
        self.lmns = col.lmns
        self.mboz = col.max_harmonic+1
        self.mnmax = col.max_harmonic+1
        self.mnmax_nyq = col.max_harmonic+1
        self.mpol = col.max_harmonic+1
        self.mpol_nyq = col.max_harmonic
        self.nboz = 0
        self.nfp = 1
        self.ns_in = psi_n_in.size
        self.ntor = 0
        self.ntor_nyq = 0
        self.rmnc = col.rmnc
        self.rmns = col.rmns
        self.s_in = psi_n_in
        self.toroidal_flux = psi_lcfs
        self.psi_lcfs = psi_lcfs
        self.psi_in = psi_n_in * self.psi_lcfs
        self.verbose = 2
        self.xm = col.mode_numbers
        self.xm_nyq = col.mode_numbers
        self.xn = np.zeros_like(col.mode_numbers)
        self.xn_nyq = np.zeros_like(col.mode_numbers)
        self.zmnc = col.zmnc
        self.zmns = col.zmns

        self.run()


if __name__ == "__main__":
    gfiles = get_test_equilibria_filenames()

    test_case = 5
    FILEPATH = gfiles[test_case]


    b = Booz_xform()
    s_in = np.linspace(0.01, 0.98, num=16)
    b.init_from_geqdsk(FILEPATH, psi_n_in=s_in)


    fig, axs = plt.subplots(2,3,tight_layout=True)

    axs = axs.ravel()

    psi_i = np.linspace(0, b.psi_lcfs)
    axs[0].plot(psi_i, b.g(psi_i))
    axs[0].plot(b.psi_b, b.Boozer_G, 'r+')
    axs[0].set_title("G")

    axs[1].plot(psi_i, b.I(psi_i))
    axs[1].plot(b.psi_b, b.Boozer_I, 'r+')
    axs[1].set_title("I")

    axs[2].plot(psi_i, b.iota_m(psi_i))
    axs[2].plot(b.psi_in, b.iota, 'r+')
    axs[2].set_title("iota")

    axs[3].plot(psi_i, b.q(psi_i))
    axs[3].plot(b.psi_in, 1/b.iota, 'r+')
    axs[3].set_title("q")

    axs[4].plot(psi_i, b.psi_p(psi_i))
    axs[4].set_title("psi_p")


    fig, ax = plt.subplots()
    ntheta = 100
    nphi = 200
    theta1d = np.linspace(0, 2 * np.pi, ntheta)
    phi1d = np.linspace(0, 2 * np.pi / b.nfp, nphi)
    phi, theta = np.meshgrid(phi1d, theta1d)

    B = b.mod_B_model().calculate_on_surface(0.8*b.toroidal_flux,
                                             phi=phi,
                                             theta=theta)

    ax.contourf(phi, theta, B)
    ax.set_xlabel(R"$\phi$")
    ax.set_ylabel(R"$\theta$")

    plt.show()
