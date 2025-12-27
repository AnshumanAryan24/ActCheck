from software_module import gelu

import numpy as np
import math

PRECISION = 5

def gelu_check(z):
    # Eq. 45
    erf_vec = np.vectorize(math.erf)
    return 0.5 * z * (1 + erf_vec(z / np.sqrt(2)))

def test_gelu():
    values = np.array([
        0,
        1,
        -1,
        *[np.random.normal(loc=0, scale=1) for _ in range(10)],
        5,
        -5
    ])

    actual = list(gelu_check(values).astype('float32'))
    test = [np.float32(entry) for entry in list(gelu(values, approximate=False))]

    res = [round(test[i], PRECISION) == round(actual[i], PRECISION) for i in range(len(test))]
    assert all(res)
