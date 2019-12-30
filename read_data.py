import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

def get_abstracts_data(data_filepath, n_data=None):
	data_df = pd.read_excel(data_filepath)
	abstracts_df = data_df['Abstract']
	if n_data != None:
		abstracts_df = abstracts_df.sample(n=n_data)
		abstracts = abstracts_df.tolist()
		for i in range(len(abstracts)):
			abstracts[i].lower()
			abstracts[i] = abstracts[i].strip()
			abstracts[i] = abstracts[i].replace('.','')
			abstracts[i] = abstracts[i].replace(',','')
	return abstracts

def map_words_to_inds(data_list):
	wordlist = []
	for item in data_list:
		words = item.split(' ')
		wordlist += words
	word_set = set(wordlist)
	word_to_ind = dict()
	ind_to_word = dict()
	for i, word in enumerate(word_set):
		word_to_ind[word] = i
		ind_to_word[i] = word
	return word_to_ind, ind_to_word

def wordlist_to_abstracts(wordlist): 
	ret_data_list = []
	line = ''
	for word in wordlist:
		if word == '*':
			ret_data_list.append(line.strip())
			line = ''
		else:
			line += word + ' '
	return ret_data_list 

def remove_less_frequent_words(data_list, min_freq=2):
	wordlist = []
	for item in data_list:
		words = item.split(' ')
		words.append('*')
		wordlist += words
	word_set = set(wordlist)
	freq_dict = dict()
	for word in word_set:
		freq_dict[word] = 0
	for word in wordlist:
		freq_dict[word] += 1
	print('start removing')
	for i in range(len(wordlist)-1, -1, -1):
		word = wordlist[i]
		freq = freq_dict[word]
		if freq < min_freq:
			wordlist.pop(i)
	print('words remoced')
	ret_data_list = wordlist_to_abstracts(wordlist)
	return ret_data_list

#create function to remove stop word_set
def rem_stop_words(list_of_abstracts):
	stop_words = set(stopwords.words('english'))
	filtered_sentence = []

	for abstract in list_of_abstracts:

		word_tokens = word_tokenize(abstract)
		for word in word_tokens:
			if word not in stop_words:
				filtered_sentence.append(word)
		filtered_sentence.append('*')
	ret_data_list = wordlist_to_abstracts(filtered_sentence)
	return ret_data_list

def create_csv_data(n_data=10):

	data_filepath = 'WebOfScience/Meta-data/Data.xlsx'
	abstracts = get_abstracts_data(data_filepath, n_data=n_data)

	abstracts = remove_less_frequent_words(abstracts)
	abstracts = rem_stop_words(abstracts)

	csv_file = open('preprocessed_abstracts_data_test.csv', 'w')
	for i, abstract in enumerate(abstracts): 
		line = str(i) + ',' + abstract + '\n'
		csv_file.write(line)
	csv_file.close()
	return abstracts

def onehot_encoder(): 
	data_filepath = 'preprocessed_abstracts_data.csv'
	abstracts_df = pd.read_csv(data_filepath)
	abstracts_df.columns = ['idx', 'abstracts']
	abstracts_df = abstracts_df['abstracts']
	abstracts = abstracts_df.tolist()

	word_to_ind, ind_to_word = map_words_to_inds(abstracts)
	onehots = []
	for abstract in abstracts: 
		abs_words = abstract.split(' ')
		rows = len(word_to_ind)
		cols = len(abs_words)
		onehot = np.zeros((rows, cols))
		for i, word in enumerate(abs_words): 
			ind = word_to_ind[word]
			onehot[ind, i] = 1
		onehots.append(onehot)
	return onehots
	
if __name__ == '__main__':
	onehots = onehot_encoder()
	
