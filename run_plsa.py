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
import matplotlib.pyplot as plt

 

    
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

def train_plsa(n_topics):
    print("Reading data")
    csv_file = '2200preprocessed_abstracts_data.csv'
    
    print("Starting training.")
    piplin = Pipeline(*DEFAULT_PIPELINE)
    piplin
    Korpus = Corpus.from_csv(csv_file, piplin, max_docs=200)
    n_topics = n_topics
    plsa = PLSA(Korpus, n_topics, True)   

    result = plsa.fit()
    plsa
    

    result.topic
    
    #testdocs = [['test', 'polyketide', 'involves', 'faster', 'lime', 'word', 'scientific', 'patient'], ['patient', 'and', 'here','another','with','some','more','word','medical']] #some text? the testset

    print("Training finished.")
    return result, Korpus

def test_plsa(trainedmodel, corpora, n_topics, beta):
    beta = beta
    result = trainedmodel
    Korpus = corpora
    n_topics = n_topics
    testdata = 'smalltest.txt'
    eta = 0.00000001
    num_words = 0
    docprobs = []

    tg = TG(testdata)
    testdocs = tg.file_gen()


    #for n in range(2, 20):
    
    print("Calculating probabilities.")
    docindex = 0
    for doc in testdocs:
        if doc == '':
            break
        words = doc.split()
        PW = 0

        try:        
            t_d = result.predict(doc)[0]
        except ValueError:
            t_d = np.ones(n_topics)/n_topics

        #missingwords = result.predict(doc)[2] #predict är en array med tre element:
        # array med sannolikheten att doc tillhör topic i på plats i
        # antal osedda ord
        # lista med osedda ord
        w_t = result.word_given_topic
        t_w = result.topic_given_word
        t_d = result.predict(doc)[0]
        d_t = result.doc_given_topic
        t = result.topic
        p_d = Korpus.get_doc(False) 
        #pdb.set_trace()
        
        for word in words:
                #print(word)
            if word not in Korpus.index.keys():
                p_w = 0
            
            else:
                    #print("this word is in the training corpus", word)
                p_w = 0
                windex = Korpus.index.get(word)
                for traindoc in range(200):
                    for topic in range(n_topics):
                        #pdb.set_trace()
                        p_w = p_w+(w_t[topic][windex][1]*t_d[topic]**beta*p_d[traindoc])
                
                PW = np.log(p_w+eta)+PW
                
            num_words += 1
            #print("Logprob for a document!", PW)
        docprobs.append(PW)
        docindex += 1
    return docprobs, num_words



def main():
    #parser = argparse.ArgumentParser(description='Probabilistic Latent Semantic Indexing')
    #parser.add_argument('--file', '-f', type=str,  required=True) #description = "CSV-file with the training data"
    #parser.add_argument('--test', '-t', type=str,  required=True) #"txt-file with the test data"
    
    #arguments = parser.parse_args()
    beta = np.linspace(0.00001, 1, 10)
    eta = 0.000000001
    total_words = 0

    docprobs = [] #the document logprobability

    perpplot = np.zeros(18)
    n_topics = 2

    result, Korpus = train_plsa(n_topics)
    perps = []
    for betav in beta:
        docprobs, num_words = test_plsa(result, Korpus, 2, betav)
        corpprob = np.sum(docprobs)
        perps.append(corpprob)
        #pdb.set_trace()


    
    print("Calculating perplexity.")
        #pdb.set_trace()
    perplexity = np.exp(-(np.max(corpprob)/num_words))
    #perpplot[n-2] = perplexity
    print("Perplexity", perplexity)
    #plt.figure()
    #plt.gcf().gca().grid(False)
    #plt.plot(np.linspace(2, 20, 18), perpplot)
    #plt.show()
 

if __name__ == "__main__":
    main()




