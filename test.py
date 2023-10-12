import numpy as np
import pandas as pd
import os
import sklearn.linear_model
import sklearn.pipeline
import text_process

data_dir = 'data_reviews'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

N, n_cols = x_train_df.shape
print("Shape of x_train_df: (%d, %d)" % (N, n_cols))
print("Shape of y_train_df: %s" % str(y_train_df.shape))

# Print out the first five rows and last five rows
tr_text_list = x_train_df['text'].values.tolist()
rows = np.arange(0, 5)
for row_id in rows:
    text = tr_text_list[row_id]
    print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id, 0], text))

print("...")
rows = np.arange(N - 5, N)
for row_id in rows:
    text = tr_text_list[row_id]
    print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id, 0], text))

tr_text_array = x_train_df['text']

# for line in tr_text_array:
#     print("\nRaw text:")
#     print(line)
#     print("Clean token list:")
#     print(tokenize_text(line))

tok_count_dict = dict()

for line in tr_text_array:
    tok_list = text_process.tokenize_text(line)
    for tok in tok_list:
        if tok in tok_count_dict:
            tok_count_dict[tok] += 1
        else:
            tok_count_dict[tok] = 1

sorted_tokens = list(sorted(tok_count_dict, key=tok_count_dict.get, reverse=True))

# for w in sorted_tokens[:10]:
#     print("%5d %s" % (tok_count_dict[w], w))
#
# for w in sorted_tokens[-10:]:
#     print("%5d %s" % (tok_count_dict[w], w))

# vocab_list = [w for w in sorted_tokens if tok_count_dict[w] >= 4]

# for w in vocab_list:
#     print("%5d %s" % (tok_count_dict[w], w))

vocab_list = ['good', 'great', 'like', 'bad', 'best', 'love', 'excellent', 'better', 'recommend',
              'nice', 'disappointed', 'pretty', 'worst', 'waste', 'amazing', 'terrible', 'wonderful',
              'poor', 'friendly', 'loved', 'delicious', 'horrible', 'cool', 'happy', 'awesome', 'awful',
              'stupid', 'perfect', 'impressed', 'comfortable', 'fantastic', 'beautiful', 'interesting',
              'perfectly', 'disappointing', 'super', 'fast', 'problem', 'bland', 'worse', 'enjoyed', 'fresh',
              'avoid', 'incredible', 'didn\'t work', 'weird', 'useless']

vocab_dict = dict()

for vocab_id, tok in enumerate(vocab_list):
    vocab_dict[tok] = vocab_id







# print(transform_text_into_feature_vector("the was this the of a an of a", vocab_dict))

# for line in tr_text_array:
#     print("\nRaw text:")
#     print(line)
#     print("Clean token list:")
#     print(transform_text_into_feature_vector(line, vocab_dict))

V = len(vocab_dict)
x_tr_NV = np.zeros((N, V))

for nn, raw_text_line in enumerate(tr_text_array):
    x_tr_NV[nn] = text_process.transform_text_into_feature_vector(raw_text_line, vocab_dict)

# print(x_tr_NV.shape)
#
# print(np.sum(x_tr_NV[:,0]))

y_true = np.ravel(y_train_df)


clf = sklearn.linear_model.LogisticRegression(
    C=1000.0, max_iter=1000)

clf.fit(x_tr_NV, y_true)

yhat_tr_N = clf.predict(x_tr_NV)
acc = np.mean( y_true == yhat_tr_N )

print("Training accuracy: %.3f" % acc)


weights_V = clf.coef_[0]
sorted_tok_ids_V = np.argsort(weights_V)

for vv in sorted_tok_ids_V:
    print("% 7.3f %s" % (weights_V[vv], vocab_list[vv]))