import pandas as pd

from gensim.models import Word2Vec

import sys
sys.path.append("..")
import config   # Read in hyperparameters from config so that we can control for input size in networks later

# Tokenizes 
def tokenize_strings(string):
	return string.split()

def main():
	# Columns: ['data', 'filenames', 'target_names', 'target', 'source', 'partition']
	data = pd.read_pickle('../data/newsgroup.pkl')
	print(config.EMBEDDING_SIZE)

if __name__ == "__main__":
	main()