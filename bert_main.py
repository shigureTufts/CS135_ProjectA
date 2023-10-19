import os
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.pipeline
import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.linear_model import SGDClassifier
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
#########################################################################################################
x_train_embed = np.load('data_reviews/x_train_BERT_embeddings.npy')
print(x_train_embed.shape)
x_test_embed = np.load('data_reviews/x_test_BERT_embeddings.npy')
#########################################################################################################




my_tfidf_classifier_pipeline = sklearn.pipeline.Pipeline([
    ('my_tfidf_feature_extractor', TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(2, 4),
                                                   vocabulary=vocab_dict, lowercase=False)),
    ('my_classifier', SGDClassifier(penalty='l2',alpha=1e-3, random_state=42,max_iter=1000, tol=None))
])

my_parameter_grid_by_name = dict()
my_parameter_grid_by_name['my_tfidf_feature_extractor__min_df'] = [1]
my_parameter_grid_by_name['my_classifier__alpha'] = np.logspace(-6, 6, 13)
y_true = np.ravel(y_train_df)
grid_searcher = sklearn.model_selection.GridSearchCV(
    my_tfidf_classifier_pipeline,
    my_parameter_grid_by_name,
    scoring='accuracy',
    cv=10,
    refit=False,
    return_train_score=True,
    n_jobs=-1,
    error_score='raise'
)

grid_searcher.fit(tr_text_list, y_true)

gsearch_results_df = pd.DataFrame(grid_searcher.cv_results_).copy()

param_keys = ['param_my_tfidf_feature_extractor__min_df', 'param_my_classifier__alpha']

# Rearrange row order so it is easy to skim
gsearch_results_df.sort_values(param_keys, inplace=True)

var = gsearch_results_df[param_keys + ['mean_test_score', 'rank_test_score']]

# print(var)

new_pipeline = sklearn.pipeline.Pipeline([
    ('my_tfidf_feature_extractor', TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1, 2),
                                                   vocabulary=vocab_dict, binary=False)),
    ('my_classifier', SGDClassifier(penalty='l2',alpha=0.001, random_state=42,max_iter=1000))
])

new_pipeline.fit(tr_text_list, y_true)
yhat_tr_N = new_pipeline.predict(tr_text_list)
acc = np.mean(y_true == yhat_tr_N)
print(acc)

yhat_test_N = new_pipeline.predict(test_text_list)
# yhat_test_N = new_pipeline.predict(test_text_list)

# print((np.array(yhat_test_N[:,1])).reshape(600,1))


for param_name in sorted(my_parameter_grid_by_name.keys()):
    print("%s: %r" % (param_name, grid_searcher.best_params_[param_name]))


new_embed_pipeline = sklearn.pipeline.Pipeline([
    ('my_classifier', sklearn.linear_model.LogisticRegression(C=0.046416, max_iter=2000, random_state=101,
                                                              penalty='l2', solver='lbfgs'))
])

new_embed_pipeline.fit(x_train_embed, y_true)
yhat_embed_tr_N = new_embed_pipeline.predict(x_train_embed)
acc = np.mean(y_true == yhat_embed_tr_N)
print(acc)

# yhat_embed_tr_N = new_embed_pipeline.predict(x_test_embed)
# yhat_embed_tr_N = new_embed_pipeline.predict_proba(x_test_embed)
# #
# print((np.array(yhat_embed_tr_N[:,1])).reshape(600,1))


my_parameter_grid_by_name_embed = dict()
my_parameter_grid_by_name_embed['my_classifier__C'] = np.logspace(-6, 6, 13)

grid_searcher_embed = sklearn.model_selection.GridSearchCV(
    new_embed_pipeline,
    my_parameter_grid_by_name_embed,
    scoring='accuracy',
    cv=10,
    refit=False,
    return_train_score=True,
    n_jobs=-1
)

grid_searcher_embed.fit(x_train_embed, y_true)

gsearch_results_embed_df = pd.DataFrame(grid_searcher_embed.cv_results_).copy()

param_keys_3 = ['param_my_classifier__C']

# Rearrange row order so it is easy to skim
gsearch_results_embed_df.sort_values(param_keys_3, inplace=True)

var_3 = gsearch_results_embed_df[param_keys_3 + ['mean_test_score', 'rank_test_score']]

print(var_3)

for param_name in sorted(my_parameter_grid_by_name_embed.keys()):
    print("%s: %r" % (param_name, grid_searcher_embed.best_params_[param_name]))