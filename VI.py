from numpy import linalg as LA
import numpy as np
from scipy.special import digamma
from read_data import onehot_encoder
from operator import itemgetter
import pdb

def VI(alpha, beta, data, k = 10):
    M = len(data)
    #phi_0 = 1/k*np.ones([M, N, k]) # N words, M documents, k topics
    #gamma_0 = M*k
    gamma_0 = np.ones([M, k])

    for i in range(k):
        for m in range(M):
                gamma_0[m, i] = alpha[i] + len(data[m][0])/k
    
    conv = 0.0001
    #phi_new = 1/k*np.ones([M, N, k])
    gamma_new = 1000000*gamma_0
    phi = []

    #data[M][V][N_m], M dokument, V vokabulärstorlek, N_m antal ord i dokument m
    
    #beta dim = k*V, beta sannolikheten för ord V givet topic k
    #alpha parametrarna i dirichletfördelningen
    ORD = []
    
    for m in range(M):
        indx = np.argwhere(data[m] == np.amax(data[m]))
        ord = sorted(indx, key = itemgetter(1))
        ORD.append(ord)
            

    while np.abs(np.sum(gamma_new-gamma_0))>1:
        for m in range(M):
            phi_new = np.zeros([len(data[m][0]), k])
            
            for n in range(len(data[m][0])):
                for i in range(k):
                    phi_new[n, i] = beta[i, ORD[m][n][0]]*np.exp(digamma(gamma_0[m, i]))
                phi_new[n,:] = phi_new[n, :]/min(phi_new[n, :])
                phi_new[n,:] = phi_new[n, :]/sum(phi_new[n, :])
        
            gamma_new[m, :] = alpha + np.sum(phi_new, axis = 0)
            phi.append(phi_new)
            gamma_0 = gamma_new
        

    return phi, gamma_new

if __name__ == '__main__':

    data = onehot_encoder('preprocessed_abstracts_data.csv')

    number_of_topics = 10
    voc_size = len(data[0])
    number_of_documents = len(data)

    alpha = np.ones(number_of_topics)/number_of_topics
    beta = np.ones([number_of_topics, voc_size])/voc_size

    testVI = VI(alpha, beta, data, number_of_topics)
    print(testVI)