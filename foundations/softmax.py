import numpy as np
from numpy.typing import NDArray
import math


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        


        max_num = max(z)
        sum_e = 0

        for i in range(len(z)):
            z[i] = z[i] - max_num
            sum_e += np.exp(z[i])

        print(np.exp(z))
        for i in range(len(z)):
            z[i] = round(np.exp(z[i]) / sum_e, 4)


        return z
