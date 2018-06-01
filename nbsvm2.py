from sklearn.feature_extraction.text import CountVectorizer
from util import clear_review_to_words
import pandas as pd
import numpy as np

def _cal_ratio(data, clf=None, label=1):
    print('cal occurance num:%s'%label)
    features=data[data.sentiment==label]['feature'].values.tolist()
    f_count=(clf.transform(features)).sum(axis=0)+1
    dataset=pd.DataFrame({'value%s'%label:np.asarray(f_count)[0]},index=range(f_count.shape[1]))
    return dataset


def cal_train_ratio(train,grams=(1,3)):
    clf = CountVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b", ngram_range=grams)
    print('countvectorizer fit train data')
    clf.fit(train.feature.values.tolist())
    print('countvectorizer dictionary len:',len(clf.vocabulary_))
    d_ratio=pd.concat((_cal_ratio(train,clf=clf, label=1),_cal_ratio(train,clf=clf, label=0)),axis=1)
    d_ratio.loc[d_ratio.value1.isna(),'value1']=1
    d_ratio.loc[d_ratio.value0.isna(),'value0']=1
    p_sum=d_ratio.value1.sum()
    n_sum=d_ratio.value0.sum()
    d_ratio['ratio']=np.log((d_ratio.value1/p_sum)/(d_ratio.value0/n_sum))
    d_ratio.pop('value0')
    d_ratio.pop('value1')
    d_ratio['id']=d_ratio.index
    d_ratio['id_ratio'] = d_ratio.id.astype(str) + ':' + d_ratio.ratio.astype(str)
    print('ratio shpae',d_ratio.shape)
    return d_ratio,clf

def _generate_nbsvm_files(data,ratio,clf,path):
    id_ratio=ratio['id_ratio'].values
    features=clf.transform(data.feature.values)
    n_fea=features.shape[0]
    i_index=0
    with open(path, 'wb') as f:
        for i in range(0,n_fea,100):
            print('generate %s:%s'%(i,i+100))
            c_features=features[i:i+100]
            c_features=(c_features/c_features)
            c_features=np.isnan(c_features)==False
            for bool_list in c_features.tolist():
                result=id_ratio[bool_list]
                if result.any():
                    f.write(('%s'%data.iloc[i_index]['sentiment']
                             +' '.join(result)
                             +'\n').encode('utf8'))

def generate_nbsvm_fils(train,test,grams):
    print('clear reveiw to feature')
    train['feature']=train.apply(lambda x:clear_review_to_words(x['review'],iswords=False),axis=1)
    print('cal ratio')
    ratio,count_clf=cal_train_ratio(train,grams)
    path='/alidata1/admin/clq/kaggle/sentiment/data/nbsvm_train.txt'
    _generate_nbsvm_files(train,ratio,count_clf,path)
    test['feature'] = test.apply(lambda x: clear_review_to_words(x['review'], iswords=False), axis=1)
    path='/alidata1/admin/clq/kaggle/sentiment/data/nbsvm_test.txt'
    _generate_nbsvm_files(test,ratio,count_clf,path)


import pandas as pd
labeled_data=pd.read_csv('/alidata1/admin/clq/kaggle/sentiment/data/labeledTrainData.tsv',header=0,delimiter='\t',quoting=3)
test_data=pd.read_csv('/alidata1/admin/clq/kaggle/sentiment/data/testData.tsv',header=0,delimiter='\t',quoting=3)
generate_nbsvm_fils(labeled_data,test=test_data,grams=(1,3))
