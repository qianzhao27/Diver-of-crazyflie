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

model_path = "pickled_models/"

def run_single_drone_test(drone_data, use_pickled_model):

	# train a random forest classifier and get prediction accuracy
	clf = None
	X_train, y_train, X_test, y_test = drone_data.get_transformed_train_test_data()

	if use_pickled_model:
		clf = pickle.load(open(model_path + "randomforest_"+drone_data.sensor+"_"+ drone_data.drone_name + ".sav", 'rb'))
	else:
		clf = RandomForestClassifier(n_estimators=150, random_state=13)
		clf.fit(X_train, y_train)
		pickle.dump(clf, open(model_path+"randomforest_"+drone_data.sensor+"_"+ drone_data.drone_name + ".sav", 'wb'))

	test_accuracy = clf.score(X_test, y_test)
	y_pred = clf.predict(X_test)
	cf = get_confusion_matrix(y_test, y_pred)

	return test_accuracy, cf

def run_cross_drone_test(drone_x, drone_y_data, sensor):
	clf = pickle.load(open(model_path + "randomforest_"+drone_y_data.sensor+"_"+ drone_x + ".sav", 'rb'))

	_, _, X_test, y_test = drone_data.get_transformed_train_test_data()
	test_accuracy = clf.score(X_test, y_test)
	y_pred = clf.predict(X_test)
	cf = get_confusion_matrix(y_test, y_pred)

	return test_accuracy, cf