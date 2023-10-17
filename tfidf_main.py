import os
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.pipeline
import nltk
import tokenize_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

data_dir = 'data_reviews'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

N, n_cols = x_train_df.shape

tr_text_list = x_train_df['text'].values.tolist()

test_text_list = x_test_df['text'].values.tolist()

nltk.download('stopwords')
useless = stopwords.words('english')

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

filtered_words = [word for word in vocab_list_2 if not word in useless]

# print(filtered_words[1:])

vocab_dict = dict()

for vocab_id, tok in enumerate(filtered_words):
    vocab_dict[tok] = vocab_id

my_tfidf_classifier_pipeline = sklearn.pipeline.Pipeline([
    ('my_bow_feature_extractor',
     TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1, 1), vocabulary=vocab_dict, binary=False)),
    ('my_classifier',
     sklearn.linear_model.LogisticRegression(C=1.0, max_iter=10000, random_state=101, solver='lbfgs'))
])


my_parameter_grid_by_name = dict()
my_parameter_grid_by_name['my_bow_feature_extractor__min_df'] = [1, 2, 4]
my_parameter_grid_by_name['my_classifier__C'] = np.logspace(-5, 5, 11)
y_true = np.ravel(y_train_df)
grid_searcher = sklearn.model_selection.GridSearchCV(
    my_tfidf_classifier_pipeline,
    my_parameter_grid_by_name,
    scoring='accuracy',
    cv=10,
    refit=False)

grid_searcher.fit(tr_text_list, y_true)

gsearch_results_df = pd.DataFrame(grid_searcher.cv_results_).copy()

param_keys = ['param_my_bow_feature_extractor__min_df', 'param_my_classifier__C']

# Rearrange row order so it is easy to skim
gsearch_results_df.sort_values(param_keys, inplace=True)

var = gsearch_results_df[param_keys + ['split0_test_score', 'rank_test_score']]

print(var)