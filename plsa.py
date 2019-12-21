from plsa import *
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.algorithms import PLSA
import plsa.preprocessors as pre 
import codecs
import argparse
import os
import pdb

 

    
class TG(object):
    def __init__(self, data):
        self.data = data

    def dir_gen(self):
        dir_name = self.data
        fnames = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]
    
        for fname in fnames:
            korp = open(fname, encoding='utf8', errors='ignore')
            korp = korp.read().lower()
            yield korp
    




def main():
    parser = argparse.ArgumentParser(description='Probabilistic Latent Semantic Indexing')
    parser.add_argument('--file', '-f', type=str,  required=True)
    
    arguments = parser.parse_args()
    
    data = arguments.file

    tg = TG(data)
 
    piplin = Pipeline(*DEFAULT_PIPELINE)
    piplin
    Korpus = Corpus(tg.file_gen(), piplin)
    

    n_topics = 5
    plsa = PLSA(Korpus, n_topics, True)
    result = plsa.fit()
    plsa

    result.topic
    new_doc = "This is a new document I would like to get a topic for"
    topic_components, number_of_new_words, new_words = result.predict(new_doc)
    
    print()
    print('Relative topic importance in new document:', topic_components)
    print('Number of previously unseen words in new document:', number_of_new_words)
    print('Previously unseen words in new document:', new_words)

    new_doc = "This is a document som borde ha fler osedda ord but maybe some has been seen"
    topic_components, number_of_new_words, new_words = result.predict(new_doc)
    
    print('Relative topic importance in new document:', topic_components)
    print('Number of previously unseen words in new document:', number_of_new_words)
    print('Previously unseen words in new document:', new_words)

    

if __name__ == "__main__":
    main()




