import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib
from utils.classifications_utils import *
from utils.data_processing_utils import *
from utils.data_visualization_utils import *
from utils.metrics_utils import *
from utils.grid_search_utils import *
from transformers.featureGenerator import FeatureGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

project = 'project2'
model_path = "pickled_models/"

level_0_wind, level_1_wind, \
level_2_wind, level_3_wind = None, None, None, None

no_wind_sensor_data, level_1_wind_sensor_data, \
level_2_wind_sensor_data, level_3_wind_sensor_data \
= None, None, None, None

X_train, X_test, \
y_train, y_test = None, None, None, None

label_0, label_1, label_2, label_3 = None, None, None, None

drone_in_use = None
sensor = None

def set_sensor_type(s):
	global sensor
	sensor = s

def set_drone_in_use(drone):
	global drone_in_use
	drone_in_use = drone

def load_all_data():
	global level_0_wind, level_1_wind, \
			level_2_wind, level_3_wind 
	num_packets = 6

	label_0 = 0
	level_0_wind = load_data(label_0, num_packets, project, drone_in_use)
	level_0_wind = separate_data_based_on_apparatus(level_0_wind)

	label_1 = 1
	level_1_wind = load_data(label_1, num_packets, project, drone_in_use)
	level_1_wind = separate_data_based_on_apparatus(level_1_wind)

	label_2 = 2
	level_2_wind = load_data(label_2, num_packets, project, drone_in_use)
	level_2_wind = separate_data_based_on_apparatus(level_2_wind)

	label_3 = 3
	level_3_wind = load_data(label_3, num_packets, project, drone_in_use)
	level_3_wind = separate_data_based_on_apparatus(level_3_wind)

def set_sensor_data():
	global no_wind_sensor_data, level_1_wind_sensor_data, \
		level_2_wind_sensor_data, level_3_wind_sensor_data

	no_wind_sensor_data = level_0_wind[sensor]
	level_1_wind_sensor_data = level_1_wind[sensor]
	level_2_wind_sensor_data = level_2_wind[sensor]
	level_3_wind_sensor_data = level_3_wind[sensor]

def generate_labels():
	global label_0, label_1, label_2, label_3

	num_labels = no_wind_sensor_data.shape[0]
	label_0 = [0 for x in range(num_labels)]
	label_1 = [1 for x in range(num_labels)]
	label_2 = [2 for x in range(num_labels)]
	label_3 = [3 for x in range(num_labels)]

def stack_data(data):
	X = data[0].append(data[1])
	X = X.append(data[2])
	X = X.append(data[3])

	return X

def set_train_test_data():
	global X_train, y_train, X_test, y_test
	data = [no_wind_sensor_data, level_1_wind_sensor_data,
		level_2_wind_sensor_data, level_3_wind_sensor_data]

	# Split No Wind Data
	X_train_0, X_test_0, y_train_0, y_test_0 = \
	train_test_split(data[0], label_0, test_size=0.2, shuffle=False)

	# Split Level 1 Wind
	X_train_1, X_test_1, y_train_1, y_test_1 = \
	train_test_split(data[1], label_1, test_size=0.2, shuffle=False)

	# Split Level 2 Wind
	X_train_2, X_test_2, y_train_2, y_test_2 = \
	train_test_split(data[2], label_2, test_size=0.2, shuffle=False)

	# Split Level 3 Wind
	X_train_3, X_test_3, y_train_3, y_test_3 = \
	train_test_split(data[3], label_3, test_size=0.2, shuffle=False)

	# stack training data
	data_train = [X_train_0, X_train_1, X_train_2, X_train_3]
	X_train = stack_data(data_train)
	y_train = np.hstack((y_train_0, y_train_1, y_train_2, y_train_3))

	# stack test data
	data_test = [X_test_0, X_test_1, X_test_2, X_test_3]
	X_test = stack_data(data_test)
	y_test = np.hstack((y_test_0, y_test_1, y_test_2, y_test_3))

def transform_features(X, y, window_size=1, num_classes=4):
	feature_generator_train = FeatureGenerator(window_size)
	feature_generator_train.fit(X, num_classes)
	X = feature_generator_train.transform(X, sensor)
	y = adjust_label_amount(y, 4)

	return X, y

def run_test(drone_clf, target_drone, sensor):
	set_sensor_type(sensor)
	set_drone_in_use(target_drone)

	load_all_data()

	set_sensor_data()

	generate_labels()

	set_train_test_data()

	global X_test, y_test

	X_test, y_test = transform_features(X_test, y_test)

	# train a random forest classifier and get prediction accuracy

	clf = pickle.load(open(model_path + "randomforest_"+sensor+"_"+ drone_clf + ".sav", 'rb'))

	test_accuracy = clf.score(X_test, y_test)
	y_pred = clf.predict(X_test)

	# plot confusion matrix
	cf = get_confusion_matrix(y_test, y_pred)
	plot_confusion_matrix(cf)

	return test_accuracy