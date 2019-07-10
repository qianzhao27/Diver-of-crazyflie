from sklearn.base import BaseEstimator, TransformerMixin
from utils.data_processing_utils import reduce_noise_with_FFT
import pandas as pd
import numpy as np

class FeatureGenerator(BaseEstimator, TransformerMixin):
	def __init__(self, sliding_window, sensor, reduce_noise=False, k=0):
		self.sliding_window = sliding_window
		self.sensor = sensor
		self.reduce_noise = reduce_noise
		self.k = k

	def get_avg_resultant_acc(self, data):

		pwd = np.power(data, 2)
		sum_xyz = np.sum(pwd, 1)
		sqrt_xyz = np.sqrt(sum_xyz)
		sum_resultant_acc = np.sum(sqrt_xyz)
		avg_resultant_acc = sum_resultant_acc/100

		return avg_resultant_acc

	def get_binned_distribution_for_one_axis(self, data):
	    max_val = data.max()
	    min_val = data.min()
	    diff = max_val - min_val
	    bin_size = diff/10
	    
	    splits = [min_val+i*bin_size for i in range(0, 11)]
	    splits[0] -= 1
	    splits[-1] += 1
	    binned_data = pd.cut(data, splits, right=True, labels=False)

	    return np.bincount(binned_data)

	def get_binned_distribution(self, data):
		results = []
		for axis in data:
			result = self.get_binned_distribution_for_one_axis(data[axis])
			assert (result.sum() == data.shape[0])
			results.append(result)

		return np.array(results).flatten()

	def get_features(self, data):
		features = np.array([])

		mu = data.mean()
		features = np.hstack((features, np.array(mu)))

		std = data.std()
		features = np.hstack((features, np.array(std)))

		avg_resultant_acc = self.get_avg_resultant_acc(data)
		features = np.append(features, avg_resultant_acc)

		binned_distribution = self.get_binned_distribution(data)
		features = np.hstack((features, binned_distribution))

		#mean absoutle difference
		mad = data.mad()
		features = np.hstack((features, np.array(mad)))

		return features

	def generate_features(self, data):
		rows, _ = data.shape
		final_data = []

		rows_needed = int(self.sliding_window * 1000 / 10)
		if rows_needed > rows:
			print('Not enough data.')
			return None

		num_data_points = data.shape[0]
		remainder = num_data_points % rows_needed
		for i in range(num_data_points):
			if i + rows_needed <= num_data_points:
				data_in_window = data.iloc[i:(rows_needed+i), :]

				if self.reduce_noise:
					data_in_window = reduce_noise_with_FFT(data_in_window, self.k)
				transformed_features = self.get_features(data_in_window)
				final_data.append(transformed_features)
			else:
				break
		        
		if remainder / rows_needed > 0.9:
			data_in_window = data.iloc[-remainder:, :]
			if self.reduce_noise:
				data_in_window = reduce_noise_with_FFT(data_in_window, self.k)
			transformed_features = self.get_features(data_in_window)
			final_data.append(transformed_features)

		return np.array(final_data)

	def make_columns(self):
		columns = []
		features = ["mu", "std", "avg_resultant_acc", "bin", "mean_abs_difference"]
		for feat in features:
			if feat is "bin":
				for i in range(30):
					columns.append("bin_"+str(i)+"_"+self.sensor)
			elif feat is "avg_resultant_acc":
				columns.append("avg_resultant_acc_"+self.sensor)
			else:
				for axis in ["x", "y", "z"]:
					columns.append(feat+"_"+axis+"_"+self.sensor)
		return columns

	def transform(self, X):
		columns = self.make_columns()
		final_df = pd.DataFrame(data=[], columns=columns)
		#rows needed to calculate one sliding window amount of data
		#1s = 1000(ms); each row of data corresponds to 10(ms)

		for c in range(self.num_classes):
			start_idx = c*self.cut_off_number
			end_idx = (c+1)*self.cut_off_number
			transformed_data = self.generate_features(X.iloc[start_idx:end_idx, :])
			transformed_data = pd.DataFrame(data=transformed_data, columns=columns)
			final_df = final_df.append(transformed_data)

		return final_df

	# we have to set the cut off number so that we know 
	# how many data points belong to each class
	def fit(self, X, num_classes):
		self.num_classes = num_classes
		self.cut_off_number = X.shape[0]//num_classes
		return self

	