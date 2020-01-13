import numpy as np
from scipy.special import gamma, digamma
from read_data import onehot_encoder
import pdb
from operator import itemgetter
from em import em
from VI import VI

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

def perplexity(g, phi, alfa, beta, data, voc_size, num_topics):
    #data typ WORD[doc][docplats] = voc_index 
    # inte 'ettord' utan typ 1938, voc_index
    logp = 0
    num_words = 0
    for index in range(len(data)):
        logp += logp
        logp = lower_bound(g[index], phi[index], alfa, beta, data[index], voc_size, num_topics)
        num_words += len(data[index])

    perp = np.exp(-np.sum(logp)/num_words)
    #perp=np.sum(perp)
    return perp

def main():
    one_hot, word_to_ind, ind_to_word = onehot_encoder('200preprocessed_abstracts_data_test.csv')
    num_topics = 3
    voc_size = one_hot[0].shape[0]
    M = len(one_hot)

    alpha, beta, phi,gamma = em(num_topics, voc_size, one_hot)
    
    data  = []
    for m in range(M):
        indx = np.argwhere(one_hot[m] == np.amax(one_hot[m]))
        word = sorted(indx, key=itemgetter(1))
        data.append(word)


    perplx = perplexity(gamma, phi, alpha, beta, data, voc_size, num_topics)

    print("Perplexity", perplx)

if __name__ == "__main__":
    main()
