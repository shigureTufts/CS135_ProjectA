import numpy as np
import pandas as pd
import os

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


def tokenize_text(raw_text):
    ''' Transform a plain-text string into a list of tokens

    We assume that *whitespace* divides tokens.

    Args
    ----
    raw_text : string

    Returns
    -------
    list_of_tokens : list of strings
        Each element is one token in the provided text
    '''
    list_of_tokens = raw_text.split()  # split method divides on whitespace by default
    for pp in range(len(list_of_tokens)):
        cur_token = list_of_tokens[pp]
        # Remove punctuation
        for punc in ['?', '!', '_', '.', ',', '"', '/']:
            cur_token = cur_token.replace(punc, "")
        # Turn to lower case
        clean_token = cur_token.lower()
        # Replace the cleaned token into the original list
        list_of_tokens[pp] = clean_token
    return list_of_tokens


tr_text_array = x_train_df['text']

for line in tr_text_array:
    print("\nRaw text:")
    print(line)
    print("Clean token list:")
    print(tokenize_text(line))

tok_count_dict = dict()

for line in tr_text_array:
    tok_list = tokenize_text(line)
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

vocab_list = [w for w in sorted_tokens if tok_count_dict[w] >= 4]

# for w in vocab_list:
#     print("%5d %s" % (tok_count_dict[w], w))

vocab_dict = dict()
for vocab_id, tok in enumerate(vocab_list):
    vocab_dict[tok] = vocab_id


def transform_text_into_feature_vector(text, vocab_dict):
    ''' Produce count feature vector for provided text

    Args
    ----
    text : string
        A string of raw text, representing a single 'review'
    vocab_dict : dict with string keys
        If token is in vocabulary, will exist as key in the dict
        If token is not in vocabulary, will not be in the dict

    Returns
    -------
    count_V : 1D numpy array, shape (V,) = (n_vocab,)
        Count vector, indicating how often each vocab word
        appears in the provided text string
    '''
    V = len(vocab_dict.keys())
    count_V = np.zeros(V)
    for tok in tokenize_text(text):
        if tok in vocab_dict:
            vv = vocab_dict[tok]
            count_V[vv] += 1
    return count_V


print(transform_text_into_feature_vector("the was this the of a an of a", vocab_dict))
