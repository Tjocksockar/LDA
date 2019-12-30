import numpy as np
import scipy.special as sci


from read_data import *
from VI import *

def em(K, V, data):
        alpha = np.ones(K)/K
        beta = np.ones((K, V))/V

        conv = 0.0001
        diff_alpha = conv + 1
        diff_beta = conv + 1
        print('Start EM iterations...')
        counter = 0
        while diff_alpha > conv and diff_beta > conv:
                #phi, gamma = VI(alpha, beta, data, K)

                ##### JUST FOR TESTING - COMMENT OUT WHEN RUNNING
                phi = []
                M = len(data)
                for d in range(M):
                    N = data[d].shape[1]
                    phi.append(np.ones((N,K)))
                gamma = np.ones((M,K))
                ##### END TEST CODE

                print('Completed VI round ' + str(counter))
                counter +=1
                # Calculating beta
                M = len(data)
                beta_new = np.zeros((K,V))
                for i in range(K):
                        for j in range(V):
                                for d in range(M):
                                        N = data[d].shape[1]
                                        for n in range(N):
                                                beta_new[i,j] += phi[d][n,i] * data[d][j,n]
                        alpha_new = alpha - np.matmul(np.linalg.inv(hessian(alpha, M)), gradient(alpha, M, gamma))
                diff_alpha = np.linalg.norm(alpha_new-alpha)
                diff_beta = np.linalg.norm(beta_new-beta)
                alpha = alpha_new
                beta = beta_new
                print('Completed EM round ' + str(counter-1))

        return alpha, beta

def gradient(alpha, M, gamma):
        grad_alpha = np.zeros(K)
        for i in range(K):
                second_term = 0
                for d in range(M):
                        second_term += (sci.digamma(gamma[d, i]) - sci.digamma(np.sum(gamma[d, :])))
                grad_alpha[i] = M * (sci.digamma(np.sum(alpha))-sci.digamma(alpha[i])) + second_term
        return grad_alpha

def hessian(alpha, M):
        hessian_alpha = np.zeros((K, K))
        for i in range(K):
                for j in range(K):
                        if i == j:
                                delta = 1
                        else:
                                delta = 0
                        hessian_alpha[i, j] = M * (sci.polygamma(1, np.sum(alpha)) - delta*sci.polygamma(1,alpha[i]))
        return hessian_alpha

if __name__ == '__main__':
        K = 10 # number of topics i.e parameters
        data = onehot_encoder()
        data = data[0:10]
        V = data[0].shape[0] # vocabulary size

        alpha, beta = em(K, V, data)
