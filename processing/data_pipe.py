import pandas as pd
import numpy as np

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

	data = pd.concat([train, test], axis = 0)

	return data

def main():
	data = retrieve_data()

if __name__ == "__main__":
	main()