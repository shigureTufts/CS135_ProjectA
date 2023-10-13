import os
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.pipeline
import nltk
import tokenize_text
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

data_dir = 'data_reviews'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

N, n_cols = x_train_df.shape
# print("Shape of x_train_df: (%d, %d)" % (N, n_cols))
# print("Shape of y_train_df: %s" % str(y_train_df.shape))

# Print out the first five rows and last five rows
tr_text_list = x_train_df['text'].values.tolist()

test_text_list = x_test_df['text'].values.tolist()

###########################################################################################################
tok_count_dict = dict()
for line in tr_text_list:
    tok_list = tokenize_text.tokenize_text(line)
    for tok in tok_list:
        if tok in tok_count_dict:
            tok_count_dict[tok] += 1
        else:
            tok_count_dict[tok] = 1
sorted_tokens = list(sorted(tok_count_dict, key=tok_count_dict.get, reverse=True))
vocab_list_2 = [w for w in sorted_tokens if tok_count_dict[w] >= 1]
vocab_list = ['good', 'great', 'like', 'bad', 'best', 'love', 'excellent', 'better', 'recommend',
              'nice', 'disappointed', 'pretty', 'worst', 'waste', 'amazing', 'terrible', 'wonderful',
              'poor', 'friendly', 'loved', 'delicious', 'horrible', 'cool', 'happy', 'awesome', 'awful',
              'stupid', 'perfect', 'impressed', 'comfortable', 'fantastic', 'beautiful', 'interesting',
              'perfectly', 'disappointing', 'super', 'fast', 'problem', 'bland', 'worse', 'enjoyed', 'fresh',
              'avoid', 'incredible', 'didn\'t work', 'weird', 'useless', 'enjoy',
              'sucked', 'disappointment', 'unfortunately', 'mediocre', 'recommended', 'pleased', 'junk']

nltk.download('stopwords')
useless = stopwords.words('english')

# print(useless)
filtered_words = [word for word in vocab_list_2 if not word in useless]

print(filtered_words[1:])

vocab_dict = dict()

for vocab_id, tok in enumerate(filtered_words):
    vocab_dict[tok] = vocab_id

bow_preprocessor = CountVectorizer(binary=False, vocabulary=vocab_dict)

bow_preprocessor.fit(tr_text_list)

sparse_arr = bow_preprocessor.transform(tr_text_list)

print(sparse_arr.shape)

dense_arr_NV = sparse_arr.toarray()

print(dense_arr_NV.shape)

# print(dense_arr_NV)

# bow_preprocessor = CountVectorizer(ngram_range=(1,1), min_df=1, max_df=1.0, binary=False)
#
# bow_preprocessor.fit(tr_text_list)

# for term, count in list(bow_preprocessor.vocabulary_.items())[:10]:
#     print("%4d %s" % (count, term))


# print(len(bow_preprocessor.vocabulary_))

my_bow_classifier_pipeline = sklearn.pipeline.Pipeline([
    ('my_bow_feature_extractor',
     CountVectorizer(min_df=1, max_df=1.0, ngram_range=(1, 1), vocabulary=vocab_dict, binary=False)),
    ('my_classifier', sklearn.linear_model.LogisticRegression(C=1.0, max_iter=1000, random_state=101))
])

my_parameter_grid_by_name = dict()
my_parameter_grid_by_name['my_bow_feature_extractor__min_df'] = [1, 2, 4]
my_parameter_grid_by_name['my_classifier__C'] = np.logspace(-5, 5, 11)

my_scoring_metric_name = 'accuracy'
y_true = np.ravel(y_train_df)
# print(y_true)

prng = np.random.RandomState(0)

valid_ids = prng.choice(np.arange(N), size=800)

valid_indicators_N = np.zeros(N)
valid_indicators_N[valid_ids] = -1

my_splitter = sklearn.model_selection.PredefinedSplit(valid_indicators_N)

grid_searcher = sklearn.model_selection.GridSearchCV(
    my_bow_classifier_pipeline,
    my_parameter_grid_by_name,
    scoring=my_scoring_metric_name,
    cv=my_splitter,
    refit=False)

grid_searcher.fit(tr_text_list, y_true)

gsearch_results_df = pd.DataFrame(grid_searcher.cv_results_).copy()

param_keys = ['param_my_bow_feature_extractor__min_df', 'param_my_classifier__C']

# Rearrange row order so it is easy to skim
gsearch_results_df.sort_values(param_keys, inplace=True)

var = gsearch_results_df[param_keys + ['split0_test_score', 'rank_test_score']]

print(var)

# y_test_pred = np.ravel(test_text_list)
# my_bow_classifier_pipeline.fit(tr_text_list, y_true)
# result = my_bow_classifier_pipeline.predict(y_test_pred)
#
# # print(result[:, np.newaxis])


# print(yhat_tr_N[:,np.newaxis])


new_pipeline = sklearn.pipeline.Pipeline([
    ('my_bow_feature_extractor', CountVectorizer(min_df=1, max_df=1.0, ngram_range=(1, 1),
                                                 vocabulary=vocab_dict, binary=False)),
    ('my_classifier', sklearn.linear_model.LogisticRegression(C=1.0, max_iter=1000, random_state=101))
])

new_pipeline.fit(tr_text_list, y_true)
yhat_tr_N = new_pipeline.predict(tr_text_list)
acc = np.mean(y_true == yhat_tr_N)
print(acc)

yhat_test_N = new_pipeline.predict_proba(test_text_list)


float_y_test = yhat_test_N[:,1]

print(float_y_test)

print(float_y_test.T)

