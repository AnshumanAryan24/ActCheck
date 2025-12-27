from software_module import sgelu

import numpy as np
import math

PRECISION = 5

def sgelu_check(z, a=1.0):
    # Eq. 48
    erf_vec = np.vectorize(math.erf)
    return a * z * erf_vec(z / np.sqrt(2))

def test_sgelu():
    values = np.array([
        0,
        1,
        -1,
        *[np.random.normal(loc=0, scale=1) for _ in range(10)],
        5,
        -5
    ])
    
    # Test default a=1.0
    actual = list(sgelu_check(values).astype('float32'))
    test = [np.float32(entry) for entry in list(sgelu(values))]
    res = [round(test[i], PRECISION) == round(actual[i], PRECISION) for i in range(len(test))]
    assert all(res)

    # Test custom a
    a_val = 0.5
    actual_a = list(sgelu_check(values, a=a_val).astype('float32'))
    test_a = [np.float32(entry) for entry in list(sgelu(values, a=a_val))]
    res_a = [round(test_a[i], PRECISION) == round(actual_a[i], PRECISION) for i in range(len(test_a))]
    assert all(res_a)
