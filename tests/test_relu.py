from software_module import relu

import numpy as np

PRECISION = 5

def relu_check(z):
    return max(0, z)

# if __name__=='__main__':
def test_relu():
    values = np.array([
        0,
        1,
        -1,
        *[np.random.normal(loc=0, scale=1) for _ in range(10)],
        100,
        -100
    ])

    actual = list(relu_check(values).astype('float32'))
    test = [np.float32(entry) for entry in list(relu(values))]

    res = [round(test[i], PRECISION) == round(actual[i], PRECISION) for i in range(len(test))]
    assert all(res)
