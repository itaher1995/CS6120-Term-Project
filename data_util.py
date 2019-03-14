import pandas as pd
import numpy as np
import pickle
import time

from torch.utils.data import DataLoader

import config

# Loads features and puts them into tensor form
def load_data():


	# Loads Sentence vectors
	with open('data/sent_embds.pkl', 'rb') as f:
		data = pickle.load(f)

	dataloader = DataLoader(data, batch_size=config.BATCH_SIZE)	

	return dataloader


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