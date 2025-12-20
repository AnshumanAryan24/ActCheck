# Import like this:
# from software_module import exported_function_name
from software_module import binary

import numpy as np

def check_binary(x):  # Will cross-check results against this function, might not be consistent like this everywhere
    return np.where(x>=0, 1, 0)

# if __name__=='__main__':
def test_binary():
    values = np.array([
        0,
        -100,
        100,
        *[np.random.normal(loc=0, scale=1) for _ in range(10)]
    ])
    
    actual = check_binary(values)  # true values
    test = binary(values)  # values from imported function

    # When using pytest later, we will use this statement.
    assert all(test == actual)
    
    # print(test == actual)