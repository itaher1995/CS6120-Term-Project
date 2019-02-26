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


def main():
	# Columns: ['data', 'filenames', 'target_names', 'target', 'source', 'partition']
	data = pd.read_pickle('../data/newsgroup.pkl')

	temp = data.iloc[0:10]['data'].apply(tokenize_strings).tolist()
	
	create_embeddings(temp)


if __name__ == "__main__":
	main()