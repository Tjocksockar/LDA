import numpy as np 
import nltk
import math
import argparse
import codecs
from collections import defaultdict
import json

class Entropy(object):
    """
    This class reads a language model file and a test file, and computes
    the entropy and perplexity of the latter. 
    """
    def __init__(self):
        self.logProb = 0

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 1 - self.lambda1

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def compute_entropy_cumulatively(self, word):
        self.logProb = (self.logProb * self.test_words_processed 
        - self.lambda1*np.log(PLSA.p(word)+ self.lambda2)) / (self.test_words_processed + 1)
        self.test_words_processed += 1

    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>
        :param test_filename: The name of the test corpus file, plain txt.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, 'r', 'utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower()) 
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
            return True
        except IOError:
            print('Error reading testfile')
            return False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')
    
    arguments = parser.parse_args()

    entropy = Entropy()
    entropy.read_model(arguments.file)
    entropy.process_test_file(arguments.test_corpus)
    perplexity = np.exp(entropy.logProb)#/bigram_tester.test_words_processed)
    print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(entropy.test_words_processed, entropy.logProb))
    print('The perplexity is:', perplexity)

if __name__ == "__main__":
    main()
