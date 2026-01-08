from software_module import shifted_relu

import numpy as np

PRECISION = 5

def shifted_relu_check(z):
    return max(-1, z)

# if __name__=='__main__':
def test_shifter_relu():
    values = np.array([
        0,
        1,
        -1,
        *[np.random.normal(loc=0, scale=1) for _ in range(10)],
        100,
        -100
    ])

    actual = list(shifted_relu_check(values).astype('float32'))
    test = [np.float32(entry) for entry in list(shifted_relu(values))]

    res = [round(test[i], PRECISION) == round(actual[i], PRECISION) for i in range(len(test))]
    assert all(res)
