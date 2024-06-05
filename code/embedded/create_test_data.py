import math

import numpy as np
import pandas as pd

init_data = np.random.uniform(low=-1.0, high=1.0, size=(3000,))
result_data = []
for i in init_data:
    result_data.append(i ** 3)

pd.DataFrame({'init': init_data, 'result': result_data}).to_csv('test_data.csv', index=False)
