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

vocab_list_2 = ['good', 'great', 'like', 'bad', 'best', 'love', 'excellent', 'better', 'recommend',
              'nice', 'disappointed', 'pretty', 'worst', 'waste', 'amazing', 'terrible', 'wonderful',
              'poor', 'friendly', 'loved', 'delicious', 'horrible', 'cool', 'happy', 'awesome', 'awful',
              'stupid', 'perfect', 'impressed', 'comfortable', 'fantastic', 'beautiful', 'interesting',
              'perfectly', 'disappointing', 'super', 'fast', 'problem', 'bland', 'worse', 'enjoyed', 'fresh',
              'avoid', 'incredible', 'didn\'t work', 'weird', 'useless', 'enjoy', 'fresh', 'avoid', 'incredible',
              'sucked', 'incredible', 'disappointment', 'unfortunately', 'mediocre', 'recommended', 'pleased', 'junk']

mylist = [5, 3, 5, 2, 1, 6, 6, 4] # 5 & 6 are duplicate numbers.
# find the length of the list
print(len(vocab_list))
# create a set from the list
myset = set(vocab_list)
# find the length of the Python set variable myset
print(len(myset))
# compare the length and print if the list contains duplicates
if len(vocab_list) != len(myset):
    print("duplicates found in the list")
else:
    print("No duplicates found in the list")
print(vocab_list[55])
print(vocab_list[59])