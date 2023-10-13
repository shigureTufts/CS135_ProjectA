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

vocab_list = ['good', 'great', 'like', 'bad', 'best', 'love', 'excellent', 'better', 'recommend',
              'nice', 'disappointed', 'pretty', 'worst', 'waste', 'amazing', 'terrible', 'wonderful',
              'poor', 'friendly', 'loved', 'delicious', 'horrible', 'cool', 'happy', 'awesome', 'awful',
              'stupid', 'perfect', 'impressed', 'comfortable', 'fantastic', 'beautiful', 'interesting',
              'perfectly', 'disappointing', 'super', 'fast', 'problem', 'bland', 'worse', 'enjoyed', 'fresh',
              'avoid', 'incredible', 'didn\'t work', 'weird', 'useless', 'enjoy', 'fresh', 'avoid', 'incredible',
              'sucked', 'incredible', 'disappointment', 'unfortunately', 'mediocre', 'recommended', 'pleased', 'junk']

