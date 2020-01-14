import numpy as np
from numpy import asarray
from numpy import savetxt
from scipy.special import gamma, digamma
from read_data import onehot_encoder
import pdb
from operator import itemgetter
from em import em
from VI import VI
import pandas as pd

#log p(w|alfa, beta) = L(g, phi, alfa, beta) + KLD

def lower_bound(g, phi, alfa, beta, doc, voc_size, num_topics):
    #doc is a list of wordindices
    #the likelihood for one document
    #g (gamma), phi should be the phi and gamma specific for that document
    
    
    E1 = np.log(gamma(np.sum(alfa)))-np.sum(np.log(gamma(alfa)))+np.sum(
        (alfa-1)*(digamma(g)-digamma(np.sum(g))))
    E2 = np.sum(phi * (digamma(g) - digamma(np.sum(g)))) #här är dimenstionerna otydliga asssssså
    E3 = 0
    for n in range(len(doc)):
        for i in range(num_topics):
                #pdb.set_trace()
                E3 += phi[n,i]*np.log(beta[i, doc[n][0]])

    E4 = - np.log(gamma(np.sum(g))) + np.sum(np.log(gamma(g)))-np.sum((g-1)*(digamma(g)-digamma(np.sum(g))))
    
    E5 = 0
    for n in range(len(doc)):
        for i in range(num_topics):
            E5 += phi[n, i]*np.log(phi[n, i])


    LB = E1 + E2 + E3 + E4 - E5
    return LB

def MCprob(alfa, beta, doc, K):
    
    vol = 1/np.math.factorial(K)
    sampling = 100 #number of samples for monte carlo 
    MC = []
    C = gamma(np.sum(alfa))/np.prod(gamma(alfa))

    for _ in range(sampling):
        docprob = 1
        theta = np.random.dirichlet(alfa)

        for word in doc:
            logdocprob = 0
            wordprob = 0
            topicprobs = np.random.multinomial(1, theta)
            #for i in range(K):
            topic = np.nonzero(topicprobs)
            wordprob = beta[topic, word[0]]*theta[topic]
                #print(wordprob)
            #print("beta", beta[topic, word[0]])
            #print("Docprob", docprob)
            logdocprob = logdocprob + np.log(wordprob)

        #pdb.set_trace()
        MC.append(theta*np.exp(logdocprob)) #det här ska vara p(theta|alfa), inte theta

    MC_prob = vol*np.sum(MC)/sampling
    #print("MCprob", MC_prob) 
    return MC_prob 

def perplexity(alfa, beta, data, voc_size, num_topics):
    #data typ WORD[doc][docplats] = voc_index 
    # inte 'ettord' utan typ 1938, voc_index
    logarray=[]
    #logp = 0
    num_words = 0
    for index in range(len(data)):
        
        #logp = lower_bound(g[index], phi[index], alfa, beta, data[index], voc_size, num_topics)
        logp = np.log(MCprob(alfa, beta, data[index], num_topics))
        #logp += logp
        logarray.append(logp)

        num_words += len(data[index])
    
    pdb.set_trace()

    perp = np.exp(-np.sum(logarray)/num_words)
    #perp=np.sum(perp)
    return perp

def main():
    data, word_to_ind, ind_to_word = onehot_encoder('2200preprocessed_abstracts_data.csv')
    train=data[0:2000]
    test=data[2000:2100]
    num_topics = 2
    voc_size = data[0].shape[0]
    M = len(test)

    #alpha, beta, phi,gamma = em(num_topics, voc_size, train)
    
    testdata  = []
    for m in range(M):
        indx = np.argwhere(test[m] == np.amax(test[m]))
        word = sorted(indx, key=itemgetter(1))
        testdata.append(word)

    alfa = pd.read_csv('2topics_alpha.csv')
    #pdb.set_trace()
    beta = pd.read_csv('2topics_beta.csv')
    beta = beta.to_numpy()
    #pdb.set_trace()

    alfa = alfa['0'].values.tolist()


    perplx = perplexity(alfa, beta, testdata, voc_size, 2)
    print("Perplexity", perplx)

if __name__ == "__main__":
    main()