import numpy as np
from numpy import asarray
from numpy import savetxt
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
    logarray=[]
    logp = 0
    num_words = 0
    for index in range(len(data)):
        logp += logp
        logp = lower_bound(g[index], phi[index], alfa, beta, data[index], voc_size, num_topics)

        if logp>-1000000000000000000:
            logarray.append(logp)
            num_words += len(data[index])

    perp = np.exp(-np.sum(logarray)/num_words)
    #perp=np.sum(perp)
    return perp

def main():
    data, word_to_ind, ind_to_word = onehot_encoder('2200preprocessed_abstracts_data.csv')
    train=data[0:2000]
    test=data[2000:2200]
    num_topics =15
    voc_size = data[0].shape[0]
    M = len(test)

    alpha, beta, phi,gamma = em(num_topics, voc_size, train)
    
    testdata  = []
    for m in range(M):
        indx = np.argwhere(test[m] == np.amax(test[m]))
        word = sorted(indx, key=itemgetter(1))
        testdata.append(word)


    #perplx = perplexity(gamma, phi, alpha, beta, testdata, voc_size, num_topics)
    csv_file_alpha = open(str(num_topics) + 'topics_alpha.csv', 'w')
    savetxt(csv_file_alpha,alpha, delimiter=',')
    #csv_file.write(alpha)
    csv_file.close()
    csv_file = open(str(num_topics) + 'topics_beta.csv', 'w')
    savetxt(csv_file_beta, beta, delimiter=',')
    csv_file.close()
    csv_file = open(str(num_topics) + 'topics_gamma.csv', 'w')
    savetxt(csv_file_gamma, gamma, delimiter=',')
    csv_file.close()
    csv_file = open(str(num_topics) + 'topics_phi.csv', 'w')
    savetxt(csv_file_phi, phi, delimiter=',')
    csv_file.close()

    #print("Perplexity", perplx)

if __name__ == "__main__":
    main()
