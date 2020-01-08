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
    parser.add_argument('--file', '-f', type=str,  required=True) #description = "CSV-file with the training data"
    parser.add_argument('--test', '-t', type=str,  required=True) #"txt-file with the test data"
    
    arguments = parser.parse_args()
    
    csv_file = arguments.file
    testdata = arguments.test

    tg = TG(testdata)
    testdocs = tg.file_gen()
    
    piplin = Pipeline(*DEFAULT_PIPELINE)
    piplin
    Korpus = Corpus.from_csv(csv_file, piplin)

    n_topics = 5
    plsa = PLSA(Korpus, n_topics, True)   

    result = plsa.fit()
    plsa

    result.topic
    
    eta = 0.00000001
    num_words = 0
    docprobs = [] #the document logprobability
    #testdocs = [['test', 'polyketide', 'involves', 'faster', 'lime', 'word', 'scientific', 'patient'], ['patient', 'and', 'here','another','with','some','more','word','medical']] #some text? the testset

    for doc in testdocs:
        words = doc.split()
        PW = 0        
        t_d = result.predict(doc)[0]
        missingwords = result.predict(doc)[2] #predict är en array med tre element:
        # array med sannolikheten att doc tillhör topic i på plats i
        # antal osedda ord
        # lista med osedda ord
        w_t = result.word_given_topic
        t_w = result.topic_given_word
        
        for word in words:
            #print(word)
            if word not in Korpus.index.keys():
                p_w = 0
            
            else:
                #print("this word is in the training corpus", word)
                p_w = 0
                windex = Korpus.index.get(word)

                for topic in range(n_topics):
                
                    p_w = p_w+(w_t[topic][windex][1]*t_d[topic]/(t_w[topic][windex]+eta))
                 
            
            PW = np.log(p_w+eta)+PW
                
            num_words += 1
        print("Logprob for a document!", PW)
        docprobs.append(PW)

    perplexity = np.exp(-(np.sum(docprobs)/num_words))
    print("Perplexity", perplexity)
 

if __name__ == "__main__":
    main()




