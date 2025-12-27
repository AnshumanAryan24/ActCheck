from software_module import calu

import numpy as np

PRECISION = 5

def calu_check(z):
    # Eq. 49
    return z * (np.arctan(z) / np.pi + 0.5)

def test_calu():
    values = np.array([
        0,
        1,
        -1,
        *[np.random.normal(loc=0, scale=1) for _ in range(10)],
        5,
        -5
    ])

    actual = list(calu_check(values).astype('float32'))
    test = [np.float32(entry) for entry in list(calu(values))]

    res = [round(test[i], PRECISION) == round(actual[i], PRECISION) for i in range(len(test))]
    assert all(res)
