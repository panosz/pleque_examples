"""
Calculate grad psi using PLEQUE
"""
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
from pleque.io.readers import read_geqdsk
from pleque.tests.utils import get_test_equilibria_filenames, load_testing_equilibrium
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import CubicSpline


gfiles = get_test_equilibria_filenames()

test_case = 5
FILEPATH = gfiles[test_case]

# Create an instance of the `Equilibrium` class
eq = read_geqdsk(FILEPATH)


#Find the Psi_N where the safety factor is 5/3
psi_onq = brentq(lambda psi_n: np.abs(eq.q(psi_n)) - 5/3, 0, 0.95)
print(r'Psi_N = {:.3f}'.format(psi_onq))

#Define the resonant flux surface using this Psi_N
surf = eq._flux_surface(psi_n=psi_onq)[0]


R = surf.R[::100]
Z = surf.Z[::100]

dpsi_dR = eq._spl_psi(R, Z, dx=1, grid=False)
dpsi_dZ = eq._spl_psi(R, Z, dy=1, grid=False)


plt.quiver(R, Z, dpsi_dR, dpsi_dZ)
plt.plot(surf.R, surf.Z)
plt.gca().set_aspect("equal")
plt.show()

