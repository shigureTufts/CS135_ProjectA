import numpy as np

print(np.logspace(-5, 5, 11))
N = 400
prng = np.random.RandomState(0)

valid_ids = prng.choice(np.arange(N), size=100)
print(valid_ids)
valid_indicators_N = np.zeros(N)
print(valid_indicators_N)
valid_indicators_N[valid_ids] = -1
print(valid_indicators_N)