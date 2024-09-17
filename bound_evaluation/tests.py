import numpy as np
import scipy

def test_if_hankel():
    """
    Verifies if the matrix is hankel. 
    1. Generate basket for every value in [1, 2*N - 1]
    2. Take a set (i, j)
    3. Put beta[i, j] in i + j basket
    4. Exhaust i and j values
    """
