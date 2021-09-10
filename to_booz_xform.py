"""
Calculate the harmonics of various quantities on multiple flux surface
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

    s_in = np.linspace(0.1, 0.8, num=16)

    iota = 1/eq.q(s_in)

    col = CollectionOfSurfaceHarmonics.from_eq(s_in, eq, 7)

    b = Booz_xform()

    b.asym = True
    b.bmnc = col.bmnc
    b.bmns = col.bmns
    b.bsubumnc = col.bsubumnc
    b.bsubumns = col.bsubumns
    b.compute_surfs = np.arange(s_in.size)
    b.iota = iota
    b.lmnc = col.lmnc
    b.lmns = col.lmns
    b.mboz = col.max_harmonic
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
    b.toroidal_flux = None
    b.verbose = True
    b.xm = col.mode_numbers
    b.xm_nyq = col.mode_numbers
    b.xn = np.zeros_like(col.mode_numbers)
    b.xn_nyq = np.zeros_like(col.mode_numbers)
    b.zmnc = col.zmnc
    b.zmns = col.zmns



    #  b.run()

