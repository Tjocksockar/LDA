from numpy import linalg as LA
import numpy as np
from scipy.special import digamma
from read_data import onehot_encoder
from operator import itemgetter
import pdb


def VI(alpha, beta, data, k=10):
    M = len(data)
    # phi_0 = 1/k*np.ones([M, N, k]) # N words, M documents, k topics
    # gamma_0 = M*k
    gamma_0 = np.ones([M, k])

    for i in range(k):
        for m in range(M):
            gamma_0[m, i] = alpha[i] + len(data[m][0]) / k

    # phi = 1/k*np.ones([M, N, k])
    gamma_new = 2*gamma_0.copy()
    phi = []
    # phi_new är unikt för varje dokument, phi ska fyllas med phi_new

    # beta dim = k*V, beta sannolikheten för ord V givet topic k?
    # alpha parametrarna i dirichletfördelningen
    WORD = []

    for m in range(M):
        indx = np.argwhere(data[m] == np.amax(data[m]))
        word = sorted(indx, key=itemgetter(1))
        WORD.append(word)

    # WORD[m][n][0] index för ord n i dokument m

    while LA.norm(gamma_new - gamma_0) > 1:
        print(LA.norm(gamma_new - gamma_0))
        gamma_0 = gamma_new.copy()
        #print("ett varv till!")
        

        for m in range(M):
            phi_new = np.ones([len(data[m][0]), k])/k

            for n in range(len(data[m][0])):
                phi_new[n, :] = beta[:, WORD[m][n][0]] * np.exp(digamma(gamma_0[m, :])-digamma(np.sum(gamma_0[m,:])))
                phi_new[n, :] = phi_new[n, :] / np.sum(phi_new[n, :])
        

            gamma_new[m, :] = alpha + np.sum(phi_new, axis=0)
            phi.append(phi_new)

    return phi, gamma_new


if __name__ == '__main__':
    data = onehot_encoder()

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
    
    testVI = VI(alpha, beta, data, number_of_topics)
