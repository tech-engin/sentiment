import pandas as pd
from util import clear_review_to_words
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from train_w2v import  generate_Xy_w2v
from train_tfidf import generate_Xy_tfidf,generate_Xy_bow
from train_nbsvm import generate_Xy_nbsvm
from stacking import  get_stacking_feature
from scipy import hstack
import os
import numpy as np
import json

p_path='/alidata1/admin/clq/kaggle/sentiment/data/'
if not os.path.exists(os.path.join(p_path,'tfidf_data')) or \
        not os.path.exists(os.path.join(p_path,'w2v_data')) or \
        not os.path.exists(os.path.join(p_path,'nbsvm_result_stacking')) or \
        not os.path.exists(os.path.join(p_path,'bow_result_stacking')) :
    print('load data')
    labeled_data=pd.read_csv(os.path.join(p_path,'labeledTrainData.tsv'),header=0,delimiter='\t',quoting=3)
    test_data=pd.read_csv(os.path.join(p_path,'testData.tsv'),header=0,delimiter='\t',quoting=3)
    unlabeled_data=pd.read_csv(os.path.join(p_path,'unlabeledTrainData.tsv'),header=0,delimiter='\t',quoting=3)
    print('clear review')
    labeled_data['feature']=labeled_data.apply(lambda x:clear_review_to_words(x['review'],iswords=False),axis=1)
    test_data['feature']=test_data.apply(lambda x:clear_review_to_words(x['review'],iswords=False),axis=1)
    unlabeled_data['feature']=unlabeled_data.apply(lambda x:clear_review_to_words(x['review'],iswords=False),axis=1)
    with open(os.path.join(p_path, 'train_y'), 'wb') as f:
        f.write(json.dumps(labeled_data.sentiment.values.tolist()).encode('utf8'))
    with open(os.path.join(p_path, 'test_id'), 'wb') as f:
        f.write(json.dumps(test_data.id.values.tolist()).encode('utf8'))
clf=RandomForestClassifier(n_estimators=200,max_depth=5)

if not os.path.exists(os.path.join(p_path,'w2v_result_stacking')) or \
    not os.path.exists(os.path.join(p_path, 'tfidf_w2v_result_stacking')):
    if not os.path.exists(os.path.join(p_path,'w2v_data')):
        print('w2v average vector')
        X_w2v,test_w2v,y_w2v,id_test=generate_Xy_w2v(labeled_data,test_data,model_path=os.path.join(p_path,'300fea_5min_10win_skg'))
        with open(os.path.join(p_path,'w2v_data'),'wb') as f:
            f.write(json.dumps({'x':X_w2v,'test':test_w2v,'y':y_w2v,'id':id_test.tolist()}).encode('utf8'))
    else:
        print('load exist w2v data')
        with open(os.path.join(p_path,'w2v_data')) as f:
            datas=json.loads(f.read())
            X_w2v, test_w2v, y_w2v, id_test=datas['x'],datas['test'],datas['y'],datas['id']
    if not os.path.exists(os.path.join(p_path,'w2v_result_stacking')):
        train_w2v, test_w2v = get_stacking_feature(clf, X_w2v, y_w2v, test_w2v)
        with open(os.path.join(p_path, 'w2v_result_stacking'),'wb') as f:
            f.write(json.dumps({'train': train_w2v, 'test': test_w2v}).encode('utf8'))
else:
    print('load w2v stacking feature')
    with open(os.path.join(p_path, 'w2v_result_stacking')) as f:
        datas = json.loads(f.read())
        train_w2v, test_w2v = datas['train'],datas['test']

if not os.path.exists(os.path.join(p_path, 'tfidf_result_stacking')) or \
        not os.path.exists(os.path.join(p_path, 'tfidf_w2v_result_stacking')):
    if not os.path.exists(os.path.join(p_path, 'tfidf_data')):
        print('predict tfidf')
        X_tfidf, test_tfidf, y_tfidf, id_test = generate_Xy_tfidf(labeled_data, test_data, unlabeled_data)
        with open(os.path.join(p_path, 'tfidf_data'), 'wb') as f:
            f.write(json.dumps({'x': X_tfidf.toarray().tolist(),
                                'test': test_tfidf.toarray().tolist(), 'y': y_tfidf, 'id': id_test.tolist()}).encode('utf8'))
    else:
        print('load exist w2v data')
        with open(os.path.join(p_path, 'w2v_data')) as f:
            datas = json.loads(f.read())
            X_tfidf, test_tfidf, y_tfidf, id_test = datas['x'], datas['test'], datas['y'], datas['id']
    if not os.path.exists(os.path.join(p_path, 'tfidf_result_stacking')):
        train_tfidf, test_tfidf = get_stacking_feature(clf, X_tfidf, y_tfidf, test_tfidf)
        with open(os.path.join(p_path, 'tfidf_result_stacking'), 'wb') as f:
            f.write(json.dumps({'train': train_tfidf, 'test': test_tfidf}).encode('utf8'))
else:
    print('load tfidf stacking feature')
    with open(os.path.join(p_path, 'tfidf_result_stacking')) as f:
        datas = json.loads(f.read())
        train_tfidf, test_tfidf = datas['train'], datas['test']

if not os.path.exists(os.path.join(p_path,'tfidf_w2v_result_stacking')):
    if 'matrix' in type(X_tfidf):X_tfidf=X_tfidf.toarray()
    if 'matrix' in type(test_tfidf):test_tfidf=test_tfidf.toarray()
    X_tfidf_w2v=hstack([X_tfidf,X_w2v])
    test_tfidf_w2v=hstack([test_tfidf,test_w2v])
    train_tfidf_w2v, test_tfidf_w2v = get_stacking_feature(clf, X_tfidf_w2v, y_w2v, test_tfidf_w2v)
    with open(os.path.join(p_path, 'tfidf_w2v_result_stacking'),'wb') as f:
        f.write(json.dumps({'train': train_tfidf_w2v, 'test': test_tfidf_w2v}).encode('utf8'))
else:
    print('load tfidf-w2v stacking feature')
    with open(os.path.join(p_path, 'tfidf_w2v_result_stacking')) as f:
        datas = json.loads(f.read())
        train_tfidf_w2v, test_tfidf_w2v = datas['train'], datas['test']

if not os.path.exists(os.path.join(p_path,'nbsvm_result_stacking')):
    print('predict nbsvm')
    X_nbsvm,test_nbsvm,y_nbsvm,id_test=generate_Xy_nbsvm(labeled_data,test_data,
        model_path=(os.path.join(p_path,'nbsvm-train.txt'),os.path.join(p_path,'nbsvm-test.txt')))
    train_nbsvm, test_nbsvm = get_stacking_feature(clf, X_nbsvm, y_nbsvm, test_nbsvm)
    with open(os.path.join(p_path, 'nbsvm_result_stacking'),'wb') as f:
        f.write(json.dumps({'train': train_nbsvm, 'test': test_nbsvm}).encode('utf8'))
else:
    print('load nbsvm stacking feature')
    with open(os.path.join(p_path, 'nbsvm_result_stacking')) as f:
        datas=json.loads(f.read())
        train_nbsvm, test_nbsvm = datas['train'],datas['test']


if not os.path.exists(os.path.join(p_path,'bow_result_stacking')):
    print('predict bow')
    X_bow,test_bow,y_bow,id_test=generate_Xy_bow(labeled_data,test_data,unlabeled_data)
    train_bow,test_bow=get_stacking_feature(clf,X_bow,y_bow,test_bow)
    with open(os.path.join(p_path,'bow_result_stacking'),'wb') as f:
        f.write(json.dumps({'train':train_bow,'test':test_bow}).encode('utf8'))
else:
    print('load bow stacking feature')
    with open(os.path.join(p_path,'bow_result_stacking')) as f:
        datas = json.loads(f.read())
        train_bow,test_bow=datas['train'],datas['test']


train_two_X=[train_bow,train_nbsvm,train_w2v,train_tfidf,train_tfidf_w2v]
test_two_X=[test_bow,test_nbsvm,test_w2v,test_tfidf,test_tfidf_w2v]
with open(os.path.join(p_path,'train_y'))as f:
    train_y=json.loads(f.read())
with open(os.path.join(p_path, 'test_id')) as f:
    test_id=json.loads(f.read())
clf.fit(np.array(train_two_X).T,train_y)
predicts=clf.predict(np.array(test_two_X).T)
df=pd.DataFrame({'id':test_id,'sentiment':predicts})
df.to_csv(os.path.join(p_path,'outstacking.csv'),index=False,quoting=3)
print('generate out csv ')



