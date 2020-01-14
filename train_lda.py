import numpy as np
from numpy import asarray
from numpy import savetxt
from scipy.special import gamma, digamma
from read_data import onehot_encoder
import pdb
from operator import itemgetter
from em import em
from VI import VI
import pandas as pd

#log p(w|alfa, beta) = L(g, phi, alfa, beta) + KLD
def main():
    data, word_to_ind, ind_to_word = onehot_encoder('2200preprocessed_abstracts_data.csv')
    train=data[0:2000]
    test=data[2000:2100]
    num_topics = 5
    voc_size = data[0].shape[0]
    M = len(test)

    alpha, beta, phi,gamma = em(num_topics, voc_size, train)


    csv_file_alpha = open(str(num_topics) + 'topics_alpha.csv', 'w')
    savetxt(csv_file_alpha,alpha, delimiter=',')
    pd.DataFrame(alpha).to_csv(csv_file_alpha)
    csv_file_alpha.close()

    csv_file_beta = open(str(num_topics) + 'topics_beta.csv', 'w')
    savetxt(csv_file_beta, beta, delimiter=',')
    pd.DataFrame(beta).to_csv(csv_file_beta)
    csv_file_beta.close()

    csv_file_gamma = open(str(num_topics) + 'topics_gamma.csv', 'w')
    savetxt(csv_file_gamma, gamma, delimiter=',')
    pd.DataFrame(gamma).to_csv(csv_file_gamma)
    csv_file_gamma.close()

    csv_file_phi = open(str(num_topics) + 'topics_phi.csv', 'w')
    for m in range(len(train)):
        savetxt(csv_file_phi, phi[m], delimiter=',')
        pd.DataFrame(phi[m]).to_csv(csv_file_phi)
    csv_file_phi.close()

    
if __name__ == "__main__":
    main()