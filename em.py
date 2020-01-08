import numpy as np
import scipy.special as sci
import time
import sys

from read_data import *
from VI import *

def em(K, V, data):
        alpha = np.ones(K)
        beta = np.empty((K,V))
        for k in range(K):
            beta[k, :] = np.random.dirichlet(np.ones(V))
            beta[k, :] = beta[k, :] / np.sum(beta[k, :])

        conv = 0.0001
        diff_alpha = conv + 1
        diff_beta = conv + 1
        print('Start EM iterations...')
        counter = 0
        while diff_alpha > conv and diff_beta > conv:
                phi, gamma = VI(alpha, beta, data, K)

                ##### JUST FOR TESTING - COMMENT OUT WHEN RUNNING
                #phi = []
                #M = len(data)
                #for d in range(M):
                #    N = data[d].shape[1]
                #    phi.append(np.ones((N,K)))
                #gamma = np.ones((M,K))
                ##### END TEST CODE

                print('Completed VI round ' + str(counter))
                counter +=1
                # Calculating beta
                M = len(data)
                beta_new = np.zeros((K,V))
                for i in range(K):
                        for d in range(M):
                                data_matrix = data[d]
                                one_inds = np.nonzero(data_matrix)
                                n_ones = one_inds[0].shape[0]
                                for l in range(n_ones):
                                        ind_j = one_inds[0][l]
                                        ind_n = one_inds[1][l]
                                        beta_new[i,ind_j] += phi[d][ind_n,i] * data[d][ind_j,ind_n] + sys.float_info.epsilon
                        beta_new[i,:] = beta_new[i,:] / np.sum(beta_new[i,:])
                        alpha_new = alpha - np.matmul(np.linalg.inv(hessian(alpha, M)), gradient(alpha, M, gamma))
                diff_alpha = np.linalg.norm(alpha_new-alpha)
                diff_beta = np.linalg.norm(beta_new-beta)
                alpha = alpha_new
                beta = beta_new
                print('Completed EM round ' + str(counter-1))
                #print(alpha)
                #print('='*50)
                #print(beta)
                #print('='*50)
                #print(alpha)
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

def generate_document(alpha, beta,K, N=20):
    theta = np.random.dirichlet(alpha, K)
    theta = theta / np.sum(theta)
    document = np.empty(N)
    for n in range(N):
        topic = np.random.multinomial(1, theta, size=1) # First arg how many times experiment runs before returning probs
        topic = np.argmax(topic)
        word = np.random.multinomial(1, beta[topic, :], size=1)
        word = np.argmax(word)
        document[n] = word
    return document

def generate_text(document, ind_to_word):
    wordlist = []
    N = document.shape[0]
    for n in range(N):
        text = ind_to_word[document[n]]
        wordlist.append(text)
    return wordlist

def get_topwords_for_topic(topic_list, beta, ind_to_word, V, n_first=15):
    top_wordlist = []
    for topic in topic_list:
        word_rank = np.argsort(beta[topic,:])
        word_top = word_rank[V-n_first:V]
        word_top = np.flip(word_top)
        text = []
        for i in range(n_first):
            text.append(ind_to_word[word_top[i]])
        top_wordlist.append(text)
    return top_wordlist

if __name__ == '__main__':
        K = 5 # number of topics i.e parameters
        data, word_to_ind, ind_to_word = onehot_encoder('2000preprocessed_abstracts_data_test.csv')
        #data = data[0:30]
        V = data[0].shape[0] # vocabulary size

        alpha, beta = em(K, V, data)
        print('Alpha is')
        print(alpha)
        print('Beta is')
        print(beta)
        print('='*50)
        document = generate_document(alpha, beta, K)
        wordlist = generate_text(document, ind_to_word)
        #print(wordlist)

        topic_list = []
        for k in range(K):
            topic_list.append(k)
        word_toplist = get_topwords_for_topic(topic_list, beta, ind_to_word, V)
        for j in range(len(topic_list)):
            print('For topic ' + str(topic_list[j]))
            print(word_toplist[j])
            print('='*50)
