from software_module import lrelu

import numpy as np

PRECISION = 5

def lrelu_check(z, a=0.01):
    return z if z >= 0 else z*a

# if __name__=='__main__':
def test_lrelu():
    values = np.array([
        0,
        1,
        -1,
        *[np.random.normal(loc=0, scale=1) for _ in range(10)],
        100,
        -100
    ])

    actual = list(lrelu_check(values).astype('float32'))
    test = [np.float32(entry) for entry in list(lrelu(values))]

    res = [round(test[i], PRECISION) == round(actual[i], PRECISION) for i in range(len(test))]
    assert all(res)

    # Test default a=0.01
    actual = list(lrelu_check(values).astype('float32'))
    test = [np.float32(entry) for entry in list(lrelu(values))]
    res = [round(test[i], PRECISION) == round(actual[i], PRECISION) for i in range(len(test))]
    assert all(res)

    # Test custom a
    a_val = 0.005
    actual_a = list(lrelu_check(values, a=a_val).astype('float32'))
    test_a = [np.float32(entry) for entry in list(lrelu(values, a=a_val))]
    res_a = [round(test_a[i], PRECISION) == round(actual_a[i], PRECISION) for i in range(len(test_a))]
    assert all(res_a)