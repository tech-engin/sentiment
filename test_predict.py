import pandas as pd
from util import clear_review_to_words
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from train_w2v import  generate_Xy_w2v
from train_tfidf import generate_Xy_tfidf,generate_Xy_bow
from train_nbsvm import generate_Xy_nbsvm
from scipy import hstack
import os
import numpy as np

p_path='/alidata1/admin/clq/kaggle/sentiment/data/'
if not os.path.exists(os.path.join(p_path,'tfidf_result')) or \
        not os.path.exists(os.path.join(p_path,'w2v_result')) or \
        not os.path.exists(os.path.join(p_path,'nbsvm_result')) or \
        not os.path.exists(os.path.join(p_path,'bow_result')) or \
        not os.path.exists(os.path.join(p_path, 'tfidf_w2v_result')):
    print('load data')
    labeled_data=pd.read_csv(os.path.join(p_path,'labeledTrainData.tsv'),header=0,delimiter='\t',quoting=3)
    test_data=pd.read_csv(os.path.join(p_path,'testData.tsv'),header=0,delimiter='\t',quoting=3)
    unlabeled_data=pd.read_csv(os.path.join(p_path,'unlabeledTrainData.tsv'),header=0,delimiter='\t',quoting=3)
    print('clear review')
    labeled_data['feature']=labeled_data.apply(lambda x:clear_review_to_words(x['review'],iswords=False),axis=1)
    test_data['feature']=test_data.apply(lambda x:clear_review_to_words(x['review'],iswords=False),axis=1)
    unlabeled_data['feature']=unlabeled_data.apply(lambda x:clear_review_to_words(x['review'],iswords=False),axis=1)
    clf=LogisticRegression(class_weight='balanced')

if not os.path.exists(os.path.join(p_path,'bow_result')):
    print('predict bow')
    X_bow,test_bow,y_bow,id_test=generate_Xy_bow(labeled_data,test_data,unlabeled_data)
    clf.fit(X_bow,y_bow)
    predicts=clf.predict(test_bow)
    df_bow=pd.DataFrame({'id':id_test,'result':predicts})
    df_bow.to_csv(os.path.join(p_path,'bow_result'),index=False)
else:
    df_bow=pd.read_csv(os.path.join(p_path,'bow_result'))

if not os.path.exists(os.path.join(p_path,'tfidf_result')) or \
    not os.path.exists(os.path.join(p_path, 'tfidf_w2v_result')):
    print('predict tfidf')
    X_tfidf,test_tfidf,y_tfidf,id_test=generate_Xy_tfidf(labeled_data,test_data,unlabeled_data)
    clf.fit(X_tfidf,y_tfidf)
    predicts=clf.predict(test_tfidf)
    df_tfidf=pd.DataFrame({'id':id_test,'result':predicts})
    df_tfidf.to_csv(os.path.join(p_path,'tfidf_result'),index=False)
else:
    df_tfidf=pd.read_csv(os.path.join(p_path,'tfidf_result'))

if not os.path.exists(os.path.join(p_path,'w2v_result')) or \
    not os.path.exists(os.path.join(p_path, 'tfidf_w2v_result')):
    print('w2v average vector')
    X_w2v,test_w2v,y_w2v,id_test=generate_Xy_w2v(labeled_data,test_data,model_path=os.path.join(p_path,'300fea_5min_10win_skg'))
    clf.fit(X_w2v, y_w2v)
    print('w2v predict')
    predicts = clf.predict(test_w2v)
    df_w2v = pd.DataFrame({'id': id_test, 'result': predicts})
    df_w2v.to_csv(os.path.join(p_path, 'w2v_result'), index=False)
else:
    df_w2v=pd.read_csv(os.path.join(p_path,'w2v_result'))

if not os.path.exists(os.path.join(p_path,'tfidf_w2v_result')):
    X_tfidf_w2v=hstack([X_tfidf.toarray(),X_w2v])
    test_tfidf_w2v=hstack([test_tfidf.toarray(),test_w2v])
    clf.fit(X_tfidf_w2v, y_w2v)
    print('tfidf_w2v predict')
    predicts = clf.predict(test_tfidf_w2v)
    df_tfidf_w2v = pd.DataFrame({'id': id_test, 'result': predicts})
    df_tfidf_w2v.to_csv(os.path.join(p_path, 'tfidf_w2v_result'), index=False)
else:
    df_tfidf_w2v = pd.read_csv(os.path.join(p_path, 'tfidf_w2v_result'))

if not os.path.exists(os.path.join(p_path,'nbsvm_result')):
    print('predict nbsvm')
    X_nbsvm,test_nbsvm,y_nbsvm,id_test=generate_Xy_nbsvm(labeled_data,test_data,
        model_path=(os.path.join(p_path,'nbsvm-train.txt'),os.path.join(p_path,'nbsvm-test.txt')))
    clf.fit(X_nbsvm, y_nbsvm)
    predicts = clf.predict(test_nbsvm)
    df_nbsvm = pd.DataFrame({'id': id_test, 'result': predicts})
    df_nbsvm.to_csv(os.path.join(p_path, 'nbsvm_result'), index=False)
else:
    df_nbsvm=pd.read_csv(os.path.join(p_path,'nbsvm_result'))

df=pd.merge(df_tfidf,df_w2v,on='id',how='outer')
df=pd.merge(df,df_nbsvm,on='id',how='outer')
df=pd.merge(df,df_bow,on='id',how='outer')
df=pd.merge(df,df_tfidf_w2v,on='id',how='outer')
df.set_index(df.id,inplace=True)
df.pop('id')
df['sentiment']=df.apply(np.mean,axis=1)
df['sentiment']=(df['sentiment']>0.5).astype(int)
df.reset_index(inplace=True)
df[['id','sentiment']].to_csv(os.path.join(p_path,'out.csv'),index=False,quoting=3)