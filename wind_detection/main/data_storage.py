import os.path
import pandas as pd
import numpy as np
from transformers.featureGenerator import FeatureGenerator
from sklearn.model_selection import train_test_split
from utils.data_processing_utils import *

project = 'project2'
data_path = 'data/transformed_data/'
suffix_train_data = '_transformed_train.csv'
suffix_test_data = '_transformed_test.csv'
window_size = 1
num_packets = 6
test_size = 0.16665
directions = ["", "away", "left", 'right']

class DataContianer():
	def __init__(self, drone_name, sensor, num_classes, reduce_noise=False, k=0, is_directional=False):
		self.drone_name = drone_name
		self.sensor = sensor
		self.num_classes = num_classes
		self.is_directional = is_directional
		self.reduce_noise = reduce_noise
		self.k = k
		self._set_train_test_data()

	def _load_data(self, wind_level, num_packets, direction=""):
		data = load_data(wind_level, num_packets, project, self.drone_name, direction)
		data = separate_data_based_on_apparatus(data)
		return data[self.sensor]

	def _set_train_test_data(self):
		X_train, X_test = pd.DataFrame(), pd.DataFrame()
		y_train, y_test = [], []

		for label in range(self.num_classes):
			wind_level = -1
			direction = ""
			if self.is_directional:
				wind_level = 2.5
				direction = directions[label]
			else:
				wind_level = label

			X = self._load_data(wind_level, num_packets, direction)
			y = [label for x in range(X.shape[0])]		

			X_train_temp, X_test_temp, y_train_temp, y_test_temp = \
			train_test_split(X, y, test_size=test_size, shuffle=False)

			X_train = X_train.append(X_train_temp)
			y_train.append(y_train_temp)
			X_test = X_test.append(X_test_temp)
			y_test.append(y_test_temp)

		self.X_train = X_train
		self.y_train = np.array(y_train).flatten()

		self.X_test = X_test
		self.y_test = np.array(y_test).flatten()

		# transform data
		directional = '_directional' if self.is_directional else ""
		noise_reduced = '_fft_'+str(self.k) if self.reduce_noise else ""

		train_data_path = data_path+self.drone_name+"/"+\
						self.sensor+directional+noise_reduced+suffix_train_data
	
		self.X_train_transformed, self.y_train_transformed = \
		self._transform_data(self.X_train, self.y_train, train_data_path)

		test_data_path = data_path+self.drone_name+"/"+\
						self.sensor+directional+noise_reduced+suffix_test_data

		self.X_test_transformed, self.y_test_transformed = \
		self._transform_data(self.X_test, self.y_test, test_data_path)

	def _transform_data(self, X, y, file_path):
		X_transformed, y_transformed = None, None
		if os.path.isfile(file_path):
			df = pd.read_csv(file_path)
			X_transformed = df.iloc[:, :-1]
			y_transformed = df.iloc[:, -1]
		else:
			X_transformed, y_transformed = self._generate_features(X, y)
			self.save_to_file(X_transformed, y_transformed, file_path)

		return X_transformed, y_transformed

	def _generate_features(self, X, y):
		feature_generator = FeatureGenerator(window_size, self.sensor, self.reduce_noise, self.k)
		feature_generator.fit(X, self.num_classes)
		X_transformed = feature_generator.transform(X)
		y_transformed = adjust_label_amount(y, self.num_classes)

		return X_transformed, y_transformed

	def save_to_file(self, X, y, file_path):
		df = X
		df['label'] = y
		df.to_csv(file_path, index=False, sep=',')

	def get_transformed_train_test_data(self):
		return self.X_train_transformed, self.y_train_transformed, self.X_test_transformed, self.y_test_transformed