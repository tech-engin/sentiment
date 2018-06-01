import pandas as pd
import numpy as np
import re
import os
from scrapy.selector import Selector
from nltk.corpus import stopwords
import nltk.stem
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score



corpus=pd.read_csv('/alidata1/admin/clq/kaggle/sentiment/data/unlabeledTrainData.tsv',header=0,delimiter='\t',quoting=3)

def clear_to_words(row_str):
    new_str=''.join( Selector(text=row_str).xpath('.//text()').extract())
    new_str=re.sub("[^a-zA-Z]"," ",new_str)
    new_str=new_str.lower()
    stops=set(stopwords.words('english'))
    words=[i for i in new_str.split() if i not in stops]
    s = nltk.stem.SnowballStemmer('english')
    words=[s.stem(word) for word in words]
    return ' '.join(words)

if not os.path.exists('/alidata1/admin/clq/kaggle/sentiment/data/cleardata.tsv'):
    corpus=corpus.review.apply(lambda x:clear_to_words(x))
    corpus.to_csv('/alidata1/admin/clq/kaggle/sentiment/data/cleardata.tsv',index=False)

if True:
    tokenizer=CountVectorizer(analyzer='word',max_features=5000)
    features=tokenizer.fit_transform(corpus.values)
else:
    tokenizer=TfidfVectorizer(analyzer='word',max_features=5000)
    features=tokenizer.fit_transform(corpus.values)
    #clf_svd=TruncatedSVD(n_components=features.shape[1]-1)
    #clf_svd.fit(features)


train=pd.read_csv('/alidata1/admin/clq/kaggle/sentiment/data/labeledTrainData.tsv',header=0,delimiter='\t',quoting=3)
test=pd.read_csv('/alidata1/admin/clq/kaggle/sentiment/data/testData.tsv',header=0,delimiter='\t',quoting=3)

train['review']=train.review.apply(clear_to_words)
test['review']=test.review.apply(clear_to_words)
X_train=tokenizer.transform(train['review'].values).toarray()
X_test=tokenizer.transform(test['review'].values).toarray()
y_train=train['sentiment'].values
X_train,X_cv,y_train,y_cv=train_test_split(X_train,y_train,test_size=0.1)

tuning_params={
    'n_estimators':range(100,501,100),
    'max_depth':range(1,7,1),
}
#{'max_depth': 6, 'n_estimators': 200}
clf=RandomizedSearchCV(RandomForestClassifier(class_weight='balanced'),tuning_params,cv=5,scoring='roc_auc')
clf.fit(X_train,y_train)
print(accuracy_score(y_cv,clf.predict(X_cv)))
print(roc_auc_score(y_cv,clf.predict(X_cv)))
ouput=pd.DataFrame({'id':test['id'].values,'sentiment':clf.predict(X_test)})
ouput.to_csv('/alidata1/admin/clq/kaggle/sentiment/data/ouput.csv',index=False,header=True,quoting=3)