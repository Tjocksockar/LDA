import numpy as np
from scipy.special import gamma, digamma
from read_data import onehot_encoder


#log p(w|alfa, beta) = L(g, phi, alfa, beta) + KLD

def lower_bound(g, phi, alfa, beta, doc):
    #doc is a list of wordindices
    #the likelihood for one document
    #g (gamma), phi should be the phi and gamma specific for that document
    E1 = np.log(gamma(np.sum(alfa)))-np.sum(np.log(gamma(alfa)))+np.sum(
    (alfa-1)*(digamma(g)-digamma(np.sum(g)))    
    )
    E2 = np.sum(phi * (digamma(g) - digamma(np.sum(g)))) #här är dimenstionerna otydliga asssssså
    #np.sum(np.sum(phi, axis = 0), axis = 1)
    E3 = 0
    for n in range(len(doc)):
        for i in range(K):
                E3 += phi[n][i]*np.log(beta[i, doc[n]])

    E4 = - np.log(gamma(np.sum(g))) + np.sum(np.log(gamma(g)))-np.sum((gamma-1)*(digamma(g)-digamma(np.sum(g))))
    E5 = phi@np.log(phi)

    LB = E1 + E2 + E3 + E4 - E5
    return LB

def perplexity(g, phi, alfa, beta, data):
    #data typ WORD[doc][docplats] = voc_index 
    # inte 'ettord' utan typ 1938, voc_index
    logp = 0
    num_words = 0
    for index, doc in enumerate(data):
        logp += logp
        logp = lower_bound(g[index], phi[index], alfa, beta, doc)
        num_words += len(doc)
    perp = exp(-logp/num_words)
    return perp

def main():
    one_hot = onehot_encoder('523preprocessed_abstracts_data.csv')
    number_of_topics = 10
    K = number_of_topics
    voc_size = len(one_hot[0][0])
    M = len(one_hot[0])

    data = []

    for m in range(M):
        indx = np.argwhere(one_hot[m] == np.amax(one_hot[m]))
        word = sorted(indx, key=itemgetter(1))
        data.append(word)

        #WORD[m][n]
        
    data = [[0,1,2,3,4,5,6,7,8,9], [3,4,5,6,8,8,9]]
    #alpha, beta = em(number_of_topics, voc_size, data)
    #phi, gamma = VI(alpha, beta, data, number_of_topics)
    alpha = np.ones(number_of_topics) / number_of_topics
    beta = np.ones([number_of_topics, voc_size])
    for m in range(2):
        phi = np.ones([len(data[m][0]), K])/K

    gamma = np.zeros([K, 2])
    for i in range(K):
        for m in range(2):
            gamma[m, i] = alpha[i] + len(data[m][0]) / K

    for k in range(number_of_topics):
        beta[k, :] = np.random.dirichlet(np.ones(voc_size))
        for i in range (voc_size):
            if beta[k,i] == 0:
                beta[k, i] += 0.00001

    perplexity = perplexity(gamma, phi, alpha, beta, data)
    print("Perplexity", perplexity)

if __name__ == "__main__":
    main()