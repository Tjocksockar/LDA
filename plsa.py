from plsa import *
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.algorithms import PLSA
import plsa.preprocessors as pre 
from plsa.algorithms.result import *
import codecs
import argparse
import os
import pdb
import numpy as np

 

    
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
    
    def file_gen(self):    
        fname = self.data
        f = open(fname, encoding='utf8', errors='ignore')
        f = f.read()
        abstracts = f.split("\n")
        for abstract in abstracts: 
            yield abstract



def main():
    parser = argparse.ArgumentParser(description='Probabilistic Latent Semantic Indexing')
    parser.add_argument('--file', '-f', type=str,  required=True)
    
    arguments = parser.parse_args()
    
    csv_file = arguments.file

    #tg = TG(data)
    #testset = TG('')
 
    piplin = Pipeline(*DEFAULT_PIPELINE)
    piplin
    Korpus = Corpus.from_csv(csv_file, piplin)

    n_topics = 5
    plsa = PLSA(Korpus, n_topics, True)   

    
    #pdb.set_trace()
    result = plsa.fit()
    plsa

    result.topic
    
    eta = 0.00000001
    num_words = 0
    docprobs = [] #the document logprobability
    testdocs = [['test', 'text', 'special', 'words', 'scientific', 'patients'], ['and' 'heres' 'another' 'with' 'some' 'more' 'words' 'medical']] #some text? the testset

    for doc in testdocs:
        PW = 0        
        t = result.predict(doc)

        t_w = result.topic_given_doc
        t_w = t_w[windex]
        for word in doc:
            windex = Korpus.index[word]
            pdb.set_trace()
            for topic in range(n_topics):
                w_t = result.word_given_topic #topics*voc_size*2??
                w_t = w_t[n][windex][1]
            PW = np.log(w_t*t/(t_w+eta)+eta)+PW
                
            num_words += 1
        docprobs.append(PW)

    perplexity = np.exp(np.sum(docprobs)/num_words)
    print("Perplexity", perplexity)
 

if __name__ == "__main__":
    main()




