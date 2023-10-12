import os
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.pipeline
from sklearn.feature_extraction.text import CountVectorizer
import text_process

data_dir = 'data_reviews'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

N, n_cols = x_train_df.shape
print("Shape of x_train_df: (%d, %d)" % (N, n_cols))
print("Shape of y_train_df: %s" % str(y_train_df.shape))

# Print out the first five rows and last five rows
tr_text_list = x_train_df['text'].values.tolist()



vocab_list = ['good', 'great', 'like', 'bad', 'best', 'love', 'excellent', 'better', 'recommend',
              'nice', 'disappointed', 'pretty', 'worst', 'waste', 'amazing', 'terrible', 'wonderful',
              'poor', 'friendly', 'loved', 'delicious', 'horrible', 'cool', 'happy', 'awesome', 'awful',
              'stupid', 'perfect', 'impressed', 'comfortable', 'fantastic', 'beautiful', 'interesting',
              'perfectly', 'disappointing', 'super', 'fast', 'problem', 'bland', 'worse', 'enjoyed', 'fresh',
              'avoid', 'incredible', 'didn\'t work', 'weird', 'useless']

vocab_dict = dict()

for vocab_id, tok in enumerate(vocab_list):
    vocab_dict[tok] = vocab_id

bow_preprocessor = CountVectorizer(binary=False, vocabulary=vocab_dict)

bow_preprocessor.fit(tr_text_list)

sparse_arr = bow_preprocessor.transform(tr_text_list)

print(sparse_arr.shape)

dense_arr_NV = sparse_arr.toarray()

print(dense_arr_NV.shape)

print(dense_arr_NV)

bow_preprocessor = CountVectorizer(ngram_range=(1,1), min_df=1, max_df=1.0, binary=False)

bow_preprocessor.fit(tr_text_list)

for term, count in list(bow_preprocessor.vocabulary_.items())[:10]:
    print("%4d %s" % (count, term))