import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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

def map_words_to_inds(wordlist):
	word_set = set(wordlist)
	word_to_ind = dict()
	ind_to_word = dict()
	for i, word in enumerate(word_set):
		word_to_ind[word] = i
		ind_to_word[i] = word
	return word_to_ind, ind_to_word

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
	ret_data_list = []
	line = ''
	for word in wordlist:
		if word == '*':
			ret_data_list.append(line.strip())
			line = ''
		else:
			line += word + ' '
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
	return filtered_sentence


	pass

if __name__ == '__main__':
	N_DATA = 20000

	data_filepath = 'WebOfScience/Meta-data/Data.xlsx'
	abstracts = get_abstracts_data(data_filepath, n_data=N_DATA)

	abstracts = remove_less_frequent_words(abstracts)
	wordlist = rem_stop_words(abstracts)

	word_to_ind, ind_to_word = map_words_to_inds(wordlist)
	print(len(word_to_ind))
	print(len(ind_to_word))

	for i in range(10):
		print()
		print(abstracts[i])
