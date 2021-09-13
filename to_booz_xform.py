"""
Read a booz_xform equilibrium from geqdsk data.
"""
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
from covariant_b_components import read_geqdsk
from pleque.tests.utils import get_test_equilibria_filenames
from booz_xform import Booz_xform
from multiple_surf_harmonics import CollectionOfSurfaceHarmonics


if __name__ == "__main__":
    gfiles = get_test_equilibria_filenames()

    test_case = 5
    FILEPATH = gfiles[test_case]

    # Create an instance of the `Equilibrium` class
    eq = read_geqdsk(FILEPATH)

    psi_lcfs = eq.lcfs.psi[0]

    s_in = np.linspace(0.01, 0.98, num=16)

    iota = 1/eq.q(s_in)

    col = CollectionOfSurfaceHarmonics.from_eq(s_in, eq, 7)

    b = Booz_xform()

    b.asym = True
    b.bmnc = col.bmnc
    b.bmns = col.bmns
    b.bsubumnc = col.bsubumnc
    b.bsubumns = col.bsubumns
    b.bsubvmnc = col.bsubvmnc
    b.bsubvmns = col.bsubvmns
    b.compute_surfs = np.arange(s_in.size)
    b.iota = iota
    b.lmnc = col.lmnc
    b.lmns = col.lmns
    b.mboz = col.max_harmonic+1
    b.mnmax = col.max_harmonic+1
    b.mnmax_nyq = col.max_harmonic+1
    b.mpol = col.max_harmonic+1
    b.mpol_nyq = col.max_harmonic
    b.nboz = 0
    b.nfp = 1
    b.ns_in = s_in.size
    b.ntor = 0
    b.ntor_nyq = 0
    b.rmnc = col.rmnc
    b.rmns = col.rmns
    b.s_in = s_in
    b.toroidal_flux = psi_lcfs
    b.psi_lcfs = psi_lcfs
    b.psi_in = s_in * b.psi_lcfs
    b.verbose = 2
    b.xm = col.mode_numbers
    b.xm_nyq = col.mode_numbers
    b.xn = np.zeros_like(col.mode_numbers)
    b.xn_nyq = np.zeros_like(col.mode_numbers)
    b.zmnc = col.zmnc
    b.zmns = col.zmns

    b.run()

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
