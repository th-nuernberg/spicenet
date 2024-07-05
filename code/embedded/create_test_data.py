import math
import random

import numpy as np
import pandas as pd

init_data = np.random.uniform(low=0, high=33.0, size=(3000,))
result_data = []
count = 0
for i in init_data:
    result_data.append(i + 5)
    # result_data.append(i ** 3)
    count += 1

init_data = init_data / 33
result_data = np.array(result_data) / 33

pd.DataFrame({'init': init_data, 'result': result_data}).to_csv('fake_temp.csv', index=False)
