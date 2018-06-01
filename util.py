from scrapy.selector import Selector
import re
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def clear_review_to_words(raw_review,iswords=True):
    new_review = ' '.join(Selector(text=raw_review).xpath('.//text()').extract())
    new_review = re.sub('n\'t', ' not', new_review)
    new_review = re.sub('\s([1-9]\d*\.?\d*)|(0\.\d*[1-9])\s', 'numabcnum', new_review)
    new_review = re.sub('[^a-zA-z]', ' ', new_review)
    new_review=new_review.replace('\\','')
    new_review = new_review.lower()
    return new_review.split() if iswords else ' '.join(new_review.split())

def clear_review_to_sentence(raw_review,tokenizer=tokenizer):
    results=[]
    for sentence in tokenizer.tokenize(raw_review.strip()):
        if len(sentence)>0:results.append(clear_review_to_words(sentence))
    return results

