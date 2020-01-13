import numpy as np
from scipy.special import gamma, digamma
from read_data import onehot_encoder
import pdb
from em import em
from VI import VI

#log p(w|alfa, beta) = L(g, phi, alfa, beta) + KLD

def lower_bound(g, phi, alfa, beta, doc, voc_size, num_topics):
    #doc is a list of wordindices
    #the likelihood for one document
    #g (gamma), phi should be the phi and gamma specific for that document
    
    
    E1 = np.log(gamma(np.sum(alfa)))-np.sum(np.log(gamma(alfa)))+np.sum(
    (alfa-1)*(digamma(g)-digamma(np.sum(g)))    
    )
    E2 = np.sum(phi * (digamma(g) - digamma(np.sum(g)))) #här är dimenstionerna otydliga asssssså
    #np.sum(np.sum(phi, axis = 0), axis = 1)
    E3 = 0
    #pdb.set_trace()
    for n in range(len(doc)):
        for i in range(num_topics):
                E3 += phi[n][i]*np.log(beta[i, doc[n]])

    E4 = - np.log(gamma(np.sum(g))) + np.sum(np.log(gamma(g)))-np.sum((g-1)*(digamma(g)-digamma(np.sum(g))))
    
    E5 = 0
    for n in range(len(doc)):
        for i in range(num_topics):
            E5 += phi[n, i]*np.log(phi[n, i])
    
    #pdb.set_trace()


    LB = E1 + E2 + E3 + E4 - E5
    return LB

def perplexity(g, phi, alfa, beta, data, voc_size, num_topics):
    #data typ WORD[doc][docplats] = voc_index 
    # inte 'ettord' utan typ 1938, voc_index
    logp = 0
    num_words = 0
    for index, doc in enumerate(data):
        logp += logp
        logp = lower_bound(g[index], phi[index], alfa, beta, doc, voc_size, num_topics)
        num_words += len(doc)
    perp = np.exp(-logp/num_words)
    return perp

def main():
    one_hot = onehot_encoder('523preprocessed_abstracts_data.csv')
    num_topics = 3
    K = num_topics
    voc_size = len(one_hot[0][0])
    #voc_size = 10
    M = len(one_hot[0])
    #M = 2

    #data = []


        #WORD[m][n]

    #data = [[0,1,1,1,3,4,3,3,2,1,1,1], [3,4,5,6,8,8,9,5,5,5,6,7]]
    alpha, beta = em(num_topics, voc_size, one_hot)
    phi, gamma = VI(alpha, beta, one_hot, num_topics)
    
    data  = []
    for m in range(M):
        indx = np.argwhere(one_hot[m] == np.amax(one_hot[m]))
        word = sorted(indx, key=itemgetter(1))
        data.append(word)

    #alpha = np.ones(num_topics) / num_topics
    #beta = np.ones([num_topics, voc_size])
    #phi = []

    #for m in range(len(data)):
    #    phi.append(np.ones([len(data[m]), K])/K)

    #gamma = np.zeros([len(data[m]), K])
    #for i in range(K):
    #    for m in range(M):
    #        gamma[m, i] = alpha[i] + len(data[m]) / K

    #for k in range(num_topics):
    #    beta[k, :] = np.random.dirichlet(np.ones(voc_size))
    #    for i in range (voc_size):
    #        if beta[k,i] == 0:
    #            beta[k, i] += 0.00001

    perplx = perplexity(gamma, phi, alpha, beta, data, voc_size, num_topics)
    print("Perplexity", perplx)

if __name__ == "__main__":
    main()
