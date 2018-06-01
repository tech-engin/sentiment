import pandas as pd
import numpy as np
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from util import clear_review_to_words
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import gensim
test_data=pd.read_csv('/alidata1/admin/clq/kaggle/sentiment/data/testData.tsv',header=0,delimiter='\t',quoting=3)
w2v_model=gensim.models.Word2Vec('/alidata1/admin/clq/kaggle/sentiment/data/5000fea_5min_10win_5neg_skg')
words=clear_review_to_words(test_data.loc[1,'review'])
for word in words:
    print(word in w2v_model)