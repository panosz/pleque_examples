"""
This script can be used to compare the running time for the evaluation of Fourier series with and without numba compilation

After running this script with:
```
ipython -i effects_of_numba_on_fourier_series.py
```
You can compare the run times with:

    >>> %timeit _evaluate_fourier_series(Cn, Sn, theta)
    802 µs ± 4.09 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> %timeit _fast_evaluate_fourier_series(Cn, Sn, theta)
    253 µs ± 1.04 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

It can be seen that the numba optimized function runs about 3 times faster.
"""

import numpy as np
from fourier_series.fourier_series import (_evaluate_fourier_series,
                                           _fast_evaluate_fourier_series)
Cn = 1/np.linspace(1, 10, 100)
Sn = 1/np.linspace(1, 10, 100)

theta = np.linspace(0, 2*np.pi, 100)

out1 = _evaluate_fourier_series(Cn, Sn, theta)
out2 = _fast_evaluate_fourier_series(Cn, Sn, theta)

assert np.allclose(out1, out2, rtol=1e-12, atol=1e-12)
