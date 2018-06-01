from gensim.models import doc2vec
from util import clear_review_to_words
import numpy as np
import gc
def getCleanLabeledReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(clear_review_to_words(review))

    labelized = []
    for i, id_label in enumerate(reviews["id"]):
        labelized.append(doc2vec.TaggedDocument(clean_reviews[i], [id_label]))
    return labelized

def train_d2v(labeled_data,test_data,unlabeled_data):
    print("Cleaning and labeling all data sets...\n")
    train_reviews = getCleanLabeledReviews(labeled_data)
    test_reviews = getCleanLabeledReviews(test_data)
    unsup_reviews = getCleanLabeledReviews(unlabeled_data)
    del labeled_data
    del test_data
    del unlabeled_data
    gc.collect()
    model_dm_name = "%dfeatures_1minwords_10context_dm" % 300
    model_dbow_name = "%dfeatures_1minwords_10context_dbow" % 300

    model_dm = doc2vec.Doc2Vec(min_count=1, window=10, vector_size=300,sample=1e-3, workers=4)
    #model_dbow = doc2vec.Doc2Vec(min_count=1, window=10, vector_size=5000,sample=1e-3, workers=4 ,dm=0)

    all_reviews =train_reviews+test_reviews+unsup_reviews
    del train_reviews
    del test_reviews
    del unsup_reviews
    gc.collect()
    #print(all_reviews.shape)
    model_dm.build_vocab(all_reviews)
    #model_dbow.build_vocab(all_reviews)
    for epoch in range(10):
        print('train model')
        perm = np.random.permutation(len(all_reviews))

        model_dm.train([all_reviews[i] for i in perm],total_examples=model_dm.corpus_count,epochs=model_dm.epochs)
        #model_dbow.train([all_reviews[i] for i in perm])

    model_dm.save(model_dm_name)
    #model_dbow.save(model_dbow_name)
    model_dm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    #model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
