import pandas as pd
from train_w2v import train_w2v
from train_nbsvm import generate_svmlight_files

labeled_data=pd.read_csv('/alidata1/admin/clq/kaggle/sentiment/data/labeledTrainData.tsv',header=0,delimiter='\t',quoting=3)
test_data=pd.read_csv('/alidata1/admin/clq/kaggle/sentiment/data/testData.tsv',header=0,delimiter='\t',quoting=3)
unlabeled_data=pd.read_csv('/alidata1/admin/clq/kaggle/sentiment/data/unlabeledTrainData.tsv',header=0,delimiter='\t',quoting=3)
print('train nbsvm')
generate_svmlight_files(labeled_data,test_data,grams=(1,2),outfn='/alidata1/admin/clq/kaggle/sentiment/data/nbsvm')
print('train word2vec')
train_w2v(labeled_data,test_data,unlabeled_data)
