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

    conv = 0.0001
    # phi = 1/k*np.ones([M, N, k])
    gamma_new = 2 * gamma_0
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

    while LA.norm(gamma_new - gamma_0) > 1e-3:
        print(LA.norm(gamma_new - gamma_0))
        gamma_0 = gamma_new.copy()
        #print("ett varv till!")

        for m in range(M):
            phi_new = np.zeros([len(data[m][0]), k])

            for n in range(len(data[m][0])):
                for i in range(k):
                    phi_new[n, i] = beta[i, WORD[m][n][0]] * np.exp(digamma(gamma_0[m, i]))
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
    beta = np.ones([number_of_topics, voc_size]) / voc_size

    testVI = VI(alpha, beta, data, number_of_topics)
    # print(testVI)
