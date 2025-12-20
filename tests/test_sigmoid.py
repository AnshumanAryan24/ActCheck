from software_module import sigmoid

import numpy as np

PRECISION = 6

def sigmoid_check(z):
    return np.reciprocal(1 + np.exp(np.negative(z)))

# if __name__=='__main__':
def test_sigmoid():
    values = np.array([
        0,
        1,
        -1,
        *[np.random.normal(loc=0, scale=1) for _ in range(10)],
        100,
        -100
    ])

    actual = list(sigmoid_check(values).astype('float32'))
    test = [np.float32(entry) for entry in list(sigmoid(values))]

    res = [round(test[i], PRECISION) == round(actual[i], PRECISION) for i in range(len(test))]
    assert all(res)
