import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

def generate_Xy_tfidf(labeled_data,test_data,unlabeled_data):
    print('tfidf feature')
    tfidf_clf=TfidfVectorizer(analyzer='word',ngram_range=(1,2),max_features=5000)
    tfidf_clf.fit(np.concatenate((labeled_data.feature.values,test_data.feature.values,unlabeled_data.feature.values)))
    print('generate vector')
    X_tfidf=tfidf_clf.transform(labeled_data.feature.values)
    y_tfidf=labeled_data.sentiment.values
    X_test=tfidf_clf.transform(test_data.feature.values)
    id_test=test_data.id.values
    return X_tfidf,X_test,y_tfidf.tolist(),id_test

def generate_Xy_bow(labeled_data,test_data,unlabeled_data):
    print('bow feature')
    bow_clf=CountVectorizer(analyzer='word',ngram_range=(1,2),max_features=5000,binary=True)
    bow_clf.fit(np.concatenate((labeled_data.feature.values,test_data.feature.values,unlabeled_data.feature.values)))
    print('generate vector')
    X_bow=bow_clf.transform(labeled_data.feature.values)
    y_bow=labeled_data.sentiment.values
    X_test=bow_clf.transform(test_data.feature.values)
    id_test=test_data.id.values
    return X_bow,X_test,y_bow.tolist(),id_test