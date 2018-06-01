import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def get_stacking_feature(clf,X_train,y_train,X_test,n_kfold=5):
    if isinstance(X_train,list):X_train=np.array(X_train)
    if isinstance(y_train,list):y_train=np.array(y_train)
    if isinstance(X_test,list):X_test=np.array(X_test)
    off_train=np.zeros((X_train.shape[0],))
    off_test=np.zeros((X_test.shape[0],))
    off_test_skf=np.empty((n_kfold,X_test.shape[0]))
    kf=KFold(n_splits=n_kfold)
    for i,(train_index,test_index) in enumerate(kf.split(X_train)):
        kf_X_train=X_train[train_index]
        kf_X_test=X_train[test_index]
        kf_y_train=y_train[train_index]

        clf.fit(kf_X_train,kf_y_train)
        off_train[test_index]=clf.predict(kf_X_test)
        off_test_skf[i,:]=clf.predict(X_test)

    off_test[:]=off_test_skf.mean(axis=0)

    return off_train.tolist(),off_test.tolist()