from numpy import linalg as LA
import numpy as np
from scipy.special import digamma
from read_data import onehot_encoder
from operator import itemgetter
import ipdb


def VI(alpha, beta, data, k=10):
    M = len(data)
    # phi_0 = 1/k*np.ones([M, N, k]) # N words, M documents, k topics
    # gamma_0 = M*k
    gamma_0 = np.ones([M, k])
    gamma_new = np.ones([M, k])
    Mgamma_0 = np.ones([M, k])
    Mgamma_new = np.ones([M, k])
    

    for i in range(k):
        for m in range(M):
            gamma_new[m, i] = alpha[i] + (len(data[m][0]) / k)

    for i in range(k):
        for m in range(M):
            Mgamma_new[m, i] = alpha[i] + (len(data[m][0]) / k)



    # phi = 1/k*np.ones([M, N_m, k]) 
    
    phi = []
    # phi_new är unikt för varje dokument, phi ska fyllas med phi_new
    # beta dim = k*V, beta sannolikheten för ord V givet topic k?
    # alpha parametrarna i dirichletfördelningen

    WORD = []

    for m in range(M):
        #ipdb.set_trace()
        indx = np.argwhere(data[m] == np.amax(data[m]))
        word = sorted(indx, key=itemgetter(1))
        WORD.append(word)

    # WORD[m][n][0] index för ord n i dokument m

    #print("Startgamma", Mgamma_new)
    phi = []
    for m in range(M):
        phi_new = np.ones([len(data[m][0]), k])/k
        Mgamma_0 = np.ones([M, k])

        # phi_new är unikt för varje dokument, phi ska fyllas med phi_new
        # så phi[m] är phi_new för dokument m
        #N_m*k, antalet ord i dokument m * antalet topics
        #phi[m][n][k] phi för dokument m, ord n i dokument m, topic k
        
        while LA.norm(Mgamma_new - Mgamma_0) > 0.01/200:
            
            Mgamma_0 = Mgamma_new.copy()

            for n in range(len(data[m][0])):
                phi_new[n, :] = beta[:, WORD[m][n][0]] * np.exp(digamma(Mgamma_0[m, :])-digamma(np.sum(Mgamma_0[m,:])))
                phi_new[n, :] = phi_new[n, :] / np.sum(phi_new[n, :])
        
            Mgamma_new[m, :] = alpha + np.sum(phi_new, axis=0)
        phi.append(phi_new)
        #print("Convergence: ", LA.norm(Mgamma_new - Mgamma_0), m)


    return phi, Mgamma_new


if __name__ == '__main__':
    data = onehot_encoder()
    data = data[0:30]

    number_of_topics = 10
    voc_size = len(data[0])
    number_of_documents = len(data)

    alpha = np.ones(number_of_topics) / number_of_topics
    beta = np.ones([number_of_topics, voc_size])

    for k in range(number_of_topics):
        beta[k, :] = np.random.dirichlet(np.ones(voc_size))
        for i in range (voc_size):
            if beta[k,i] == 0:
                beta[k, i] = 0.00000000001
    
    slutphi, slutgamma = VI(alpha, beta, data, number_of_topics)
    print("Slutphi", slutphi)
