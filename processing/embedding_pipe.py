import pandas as pd
import numpy as np
from time import time

from gensim.models import Word2Vec

import sys
sys.path.append("..")
import config   # Read in hyperparameters from config so that we can control for input size in networks later

# Tokenizes 
def tokenize_strings(string):
	return string.split()


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

	features = [np.mean([model.wv[token] for token in text], axis = 0) for text in texts]
	
	return features


def main():
	# Columns: ['data', 'filenames', 'target_names', 'target', 'source', 'partition']
	data = pd.read_pickle('../data/newsgroup.pkl')

	# Extract tokenized text from dataframe
	texts = data['data'].apply(tokenize_strings).tolist()
	
	# Train Word2Vec model
	model = create_embeddings(texts)

	# Calculate the document vectors for each document
	doc_vectors = calculate_text_features(texts, model)

	# Add document vectors to dataframe as column 'doc_vector'
	data['doc_vector'] = doc_vectors

	# Save dataframe to new pickle file
	data.to_pickle('../data/newsgroup_vectors.pkl')


if __name__ == "__main__":
	main()