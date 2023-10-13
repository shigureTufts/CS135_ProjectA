import os
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.pipeline
from sklearn.feature_extraction.text import CountVectorizer

data_dir = 'data_reviews'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

N, n_cols = x_train_df.shape

tr_text_list = x_train_df['text'].values.tolist()

test_text_list = x_test_df['text'].values.tolist()

vocab_list = ['good', 'great', 'like', 'bad', 'best', 'love', 'excellent', 'better', 'recommend',
              'nice', 'disappointed', 'pretty', 'worst', 'waste', 'amazing', 'terrible', 'wonderful',
              'poor', 'friendly', 'loved', 'delicious', 'horrible', 'cool', 'happy', 'awesome', 'awful',
              'stupid', 'perfect', 'impressed', 'comfortable', 'fantastic', 'beautiful', 'interesting',
              'perfectly', 'disappointing', 'super', 'fast', 'problem', 'bland', 'worse', 'enjoyed', 'fresh',
              'avoid', 'incredible', 'didn\'t work', 'weird', 'useless', 'enjoy', 'fresh', 'avoid', 'incredible',
              'sucked', 'incredible', 'disappointment', 'unfortunately', 'mediocre', 'recommended', 'pleased', 'junk']

vocab_dict = dict()

for vocab_id, tok in enumerate(vocab_list):
    vocab_dict[tok] = vocab_id

###############################################################################################333

prng = np.random.RandomState(0)
valid_ids = prng.choice(np.arange(N), size=400)
valid_indicators_N = np.zeros(N)
valid_indicators_N[valid_ids] = -1
my_splitter = sklearn.model_selection.PredefinedSplit(valid_indicators_N)

my_bow_classifier_pipeline = sklearn.pipeline.Pipeline([
    ('my_bow_feature_extractor', CountVectorizer(min_df=1, max_df=1.0, ngram_range=(1, 1),
                                                 vocabulary=vocab_dict, binary=False)),
    ('my_classifier', sklearn.linear_model.LogisticRegression(C=100.0, max_iter=100, random_state=101))
])

y_true = np.ravel(y_train_df)
my_bow_classifier_pipeline.fit(tr_text_list, y_true)

yhat_tr_N = my_bow_classifier_pipeline.predict(tr_text_list)

# print(yhat_tr_N[:,np.newaxis])

acc = np.mean(y_true == yhat_tr_N)
print(acc)
