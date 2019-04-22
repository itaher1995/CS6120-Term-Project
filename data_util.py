import pandas as pd
import numpy as np
import pickle
import time

from torch.utils.data import DataLoader

import config

# Loads features and puts them into tensor form
# Takes train or test partition
def load_data(partition='train', processing='CV'):


    # Loads Sentence vectors
    if processing == 'CV':
        with open('../data/newsgroup_CV.pkl', 'rb') as f:
            data = pickle.load(f)

        # Creates sample of data
        if config.IS_SAMPLE:
            train_samp = data[data.partition == 'train'].sample(frac=config.SAMPLE_FRAC, random_state=1)
            test_samp = data[data.partition == 'test'].sample(frac=config.SAMPLE_FRAC, random_state=1)
            data = pd.concat([train_samp, test_samp], ignore_index=True).reset_index(drop=True)

        data = data[data.partition == partition]
        
        dataloader = DataLoader(data[['CV_features', 'target']].values.tolist())
        return dataloader, len(data)


    with open('../data/sent_embds.pkl', 'rb') as f:
        data = pickle.load(f)
        # Gets metadata
        metadata = pd.read_pickle('../data/newsgroup_map.pkl')

    idx_subset = metadata[metadata.partition == partition]['idx_map'].tolist()
    
    dataloader = DataLoader(np.array(data)[idx_subset].tolist(), batch_size=config.BATCH_SIZE)	

    return dataloader,len(idx_subset)


# Takens straight from odenet_mnist.py in torchdiffeq project
def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
