from gensim.models import word2vec,Word2Vec
from util import clear_review_to_sentence
import numpy as np

def train_w2v(labeled_data,test_data,unlabeled_data):
    corpus=labeled_data['review'].values.tolist()+test_data['review'].values.tolist()+unlabeled_data['review'].values.tolist()
    sentences=[]
    for review in set(corpus):
        sentences.extend(clear_review_to_sentence(review))
    print('start train word2vec')
    model=word2vec.Word2Vec(sentences,
                            size=300,
                            window=10,
                            min_count=5,
                            workers=4,
                            sample=1e-3)
    model.init_sims(replace=True)
    model_name = "/alidata1/admin/clq/kaggle/sentiment/data/300fea_5min_10win_skg"
    model.save(model_name)



def generate_Xy_w2v(labeled_data,test_data,model_path):
    model=Word2Vec.load(model_path)
    def cal_ave_vec(data):
        train_words=data.feature.values.tolist()
        train_words=[[model[word] for word in sentence.split() if word in model] for sentence in train_words]
        return [np.average(vectors,axis=0).tolist() for vectors in train_words]

    X_labeled=cal_ave_vec(labeled_data)
    y_labeled=labeled_data.sentiment.values
    X_test=cal_ave_vec(test_data)
    id_test = test_data.id.values
    return X_labeled,X_test,y_labeled.tolist(),id_test


