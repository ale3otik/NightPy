import pandas as pd 
import numpy as np

import feature_generator
import datagen_Miller

from tqdm import tqdm


def train_builder(X, feature_expander, n_samples, batch_size=100):
	datagen = datagen_Miller.DataGenerator(forecast_win=1, features_win=2, batch_size=batch_size)
	X_new = None
	n_samples /= batch_size
	for X_batch, _ in tqdm(datagen.flow(X, np.zeros(X.shape[0])), position=0):

		n_samples -= 1
		if n_samples < 0:
			break

		if X_new is None:
			X_new = X_batch
		else:
			X_new = np.concatenate((X_new, X_batch),axis=0)

	return X_new