import pandas as pd
import numpy as np
import re

from sklearn.datasets import fetch_20newsgroups


# Loads newsgroup 'train' or 'test' data and arranges into dataframe
def data_to_df(subset):
	data = fetch_20newsgroups(subset=subset)	# Get dataset from sklearn, comes as dict

	# Convert data to DataFrame
	# Leave out 'target_names' and 'DESCR' because the lengths do not match and not important data
	data_df = pd.DataFrame({k:v for k,v in data.items() if k not in ['target_names', 'DESCR']})
	
	# Join the actual source names instead of source codes so that we can discen class things
	data_df = data_df.merge(pd.DataFrame(data['target_names'], 
							columns=['source']).reset_index(),   # New newsgroup name column called 'source'
							how='left', 
							left_on = 'target', 
							right_on='index').drop('index', axis = 1)

	return data_df


def retrieve_data():
	train = data_to_df('train')
	train['partition'] = 'train'   # Add partition to merge dataframes while keeping info seperate
	
	test = data_to_df('test')
	test['partition'] = 'test'   # Add partition to merge dataframes while keeping info seperate

	data = pd.concat([train, test], axis = 0).reset_index(drop=True)

	return data


# Generalized cleaning method to be applied for each input string
def clean_string(string):
	string = re.sub('\\n(\\n)+', '\\n', string) # Remove empty lines
	string = re.sub('\\n', ' ', string) # Replace newlines with spaces
	string = re.sub(' ( )+', ' ', string) # Remove duplicate spaces
	string = string.replace('\t', '') # Remove tabs
	return string.strip()


def main():
	# Retrieves DataFrame of train and test data with the columns:
	# ['data', 'filenames', 'target_names', 'target', 'source', 'partition']
	data = retrieve_data()

	data.data = data.data.apply(clean_string)

	vectorizer = CountVectorizer(max_features = 1500)
	data['CV_features'] = vectorizer.fit_transform(data.data.tolist()).toarray().tolist()

	data.to_pickle('../data/newsgroup_CV.pkl')

if __name__ == "__main__":
	main()