import pandas as pd
import numpy as np
from time import time
import pickle

from gensim.models import Word2Vec

import sys
sys.path.append("..")
import config   # Read in hyperparameters from config so that we can control for input size in networks later

import spacy

# Make global nlp variable
nlp = spacy.load('en',disable=['ner', 'textcat', 'parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))
nlp.max_length = 4303981

# Tokenizes 
def tokenize_strings(string):
	return string.split()


# Divides documents into sentences
def get_sentences(text):
    
    tokens = nlp(text)
    
    sents = []
    for sent in tokens.sents:
        sents.append(str(sent))
        
    return sents

# Creates word embeddings using gensim Word2Vec
def create_embeddings(texts, load_pretrained=False):
	"""
	Input: list of documents, where each document is tokenized
	Output: Word embeddings for vocabulary
	"""
	start = time()

	if load_pretrained:
		# For if we want to include a pretrained vector later
		pass
	else:
		model = Word2Vec(texts, size=config.DIM_EMBEDDING, sg=1, window=5, min_count=1, workers=4)

	model.train(texts, total_examples=model.corpus_count, epochs=model.epochs)

	print('Model trained in', (time()-start)/60, 'min')

	return model


# Takes a group of texts and trained model, then calculates the document vector for each text by averageing words features
def calculate_text_features(texts, model):
	"""
	Input: group of tokenized texts and trained Word2Vec model
	Output: List of document features size len(texts) by config.DIM_EMBEDDING
	where each document feature set is of size config.DIM_EMBEDDING
	"""

	if config.PROCESSING == 'sent':
		# Processing for sentence in texts
		# Texts is slight misnomer as it is only one sentence
		features = np.mean([model.wv[token] for token in texts], axis = 0)
	else:	
		# Processing for entire list of tokenized texts
		features = [np.mean([model.wv[token] for token in text], axis = 0) for text in texts]
	
	return features

def get_doc_embeddings(docs, max_length, model):

	for d in docs:
		for s in d:
			calculate_text_features(s, model) 

	# Calculates the embeddings for each sentence
	embds = [[calculate_text_features(s, model) for s in d] for d in docs]

	# Adds on extre vectors to padd
	# Pads with 0's for a dim embedding length vector
	# Adds the number of zero vectors based on the difference between the current number of sentence vectors and the
	# max number of sentence vectors
	embds = [v + [np.zeros(config.DIM_EMBEDDING)] * (max_length - len(v)) for v in embds]

	return embds


def main():
	# Columns: ['data', 'filenames', 'target_names', 'target', 'source', 'partition']
	data = pd.read_pickle('../data/newsgroup.pkl')

	# Need a very different processing scheme to make vectors per sentence per doc
	if config.PROCESSING == 'sent':
		
		if config.LOAD_SENTS:
			with open('../data/text_sents.pkl', 'rb') as f:
				text_sents = pickle.load(f)

		else:
			start = time()
			text_sents = data['data'].apply(get_sentences).tolist()
			print('Sentences split in:' (time()-start)/60, 'min')

		# Maximium number of sentences in a document
		max_doc_sents = max([len(d) for d in text_sents])
		print('Maximium number of sentences:', max_doc_sents)

		# Tokenize sentences
		text_sents = [[tokenize_strings(s) for s in d] for d in text_sents]

		# Load a seperate embedding for sentence processing
		if config.LOAD_EMBEDDINGS:
			with open('../embeddings/sent_word_embeddigns.pkl', 'rb') as f:
				model = pickle.load(f)
		else:
			# Flattens sentences per text to get right vocab
			flat_texts = [s for d in text_sents for s in d]

			# Train Word2Vec model
			model = create_embeddings(flat_texts)

			with open('../embeddings/sent_word_embeddigns.pkl', 'wb') as f:
				pickle.dump(model, f)

		# Calculate and retrieve avereage setnence embeddings
		embds = get_doc_embeddings(text_sents, max_doc_sents, model)

		# Zips embeddings and classes for iterater later
		embds_zip = [[embds[i], data['target'].tolist()[i]] for i in range(len(embds))]

		with open('../data/sent_embds.pkl', 'wb') as f:
			pickle.dump(embds_zip, f)

		# Append index map
		data['idx_map'] = list(range(len(embds)))

		data.to_pickle('../data/newsgroup_map.pkl')


	# Standard word embedding without splitting sentences
	else:

		# Extract tokenized text from dataframe
		texts = data['data'].apply(tokenize_strings).tolist()
		
		if config.LOAD_EMBEDDINGS:
			with open('../embeddings/word_embeddigns.pkl', 'rb') as f:
				model = pickle.load(f)
		else:
			# Train Word2Vec model
			model = create_embeddings(texts)

			with open('../embeddings/word_embeddigns.pkl', 'wb') as f:
				pickle.dump(model, f)

		# Calculate the document vectors for each document
		doc_vectors = calculate_text_features(texts, model)

		# Add document vectors to dataframe as column 'doc_vector'
		data['doc_vector'] = doc_vectors

		# Save dataframe to new pickle file
		data.to_pickle('../data/newsgroup_vectors.pkl')


if __name__ == "__main__":
	main()