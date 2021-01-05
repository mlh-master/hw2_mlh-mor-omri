import pandas as pd
import numpy as np
import math
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def remove_nan(data_features):
	"""
	:param data: Pandas series of T1D features
	:return: A dictionary of clean data called clean_data
	"""
	clean_data = data_features.copy().dropna()
	return clean_data


def to_numeric(data):
	"""
	:param features: dataframe contain features series
	:return: dataframe of numeric values
	"""
	numeric_data = data.copy().replace(['Yes', 'Positive', 'Male', 'No', 'Negative', 'Female'], [1, 1, 1, 0, 0, 0])
	return numeric_data


def one_hot_vectors(data):
	"""
	:param features: dataframe contain features series
	:return: dataframe of numeric values
	"""
	if 'Age' in data.columns.values:
		data_ = data.copy()
		max_age = np.max(data['Age'])
		min_age = np.min(data['Age'])
		max_category = math.ceil(max_age / 10) * 10
		min_category = math.floor(min_age / 10) * 10
		new_categ = int((max_category - min_category) / 10 - 1) + 2
		age_categories = list(map(int, np.linspace(min_category, max_category, new_categ)))
		age_categories_str = list(map(str, age_categories))
		categories = pd.DataFrame(np.zeros([data.shape[0], new_categ]))
		categories.columns = age_categories_str
		data_ = pd.concat([data_, categories], axis=1)
		data_ = data_.drop('Age', axis=1)
		for idx, age in enumerate(data['Age']):
			fit_category = math.floor(age / 10) * 10
			data_.at[idx, f'{fit_category}'] = 1
	one_hot_vectors = data_.copy()
	# .replace(['Yes', 'Positive', 'Male', 'No', 'Negative', 'Female'], [1, 1, 1, 0, 0, 0])
	try:
		one_hot_vectors.applymap(int)
	except Exception:
		print('There is still Nan values in data...')
	return one_hot_vectors


def fill_values(features):
	"""
	:param features: dataframe contain features series
	:return: dataframe of filled values
	"""
	clean_data = features.replace(['Yes', 'No'], [1, 0]).copy()

	for feat in clean_data:
		if feat == 'Age' or feat == 'Gender':
			continue
		else:
			num_yes = clean_data.loc[:, feat].sum()
			feat_len = len(clean_data.loc[:, feat])

		idx = np.where(clean_data.loc[:, feat].isna())[0]
		s = np.random.binomial(1, num_yes / feat_len, idx.size)
		i = 0
		while i < idx.size:
			clean_data.loc[idx[i], feat] = int(s[i])
			i += 1

	return clean_data


def fix_values(data, flag='fv'):
	"""
	combine fill_values and remove_nan
	:param data: dataframe contain features series
	:param: flag : 'rn' -> remove nan, flag = 'fv' -> fill values
	:return: A df of clean data called clean_data
	"""
	clean_data = data.copy()
	if flag == 'rn':
		clean_data = clean_data.dropna()
	if flag == 'fv':
		for feat in clean_data:
			if feat == 'Age' or feat == 'Gender' or feat == 'Diagnosis':
				continue
			else:
				num_yes = clean_data.loc[:, feat].sum()
				feat_len = len(clean_data.loc[:, feat])
			idx = np.where(clean_data.loc[:, feat].isna())[0]
			s = np.random.binomial(1, num_yes / feat_len, idx.size)
			i = 0
			while i < idx.size:
				clean_data.loc[idx[i], feat] = int(s[i])
				i += 1

	t1d_features = clean_data[['Age', 'Gender', 'Increased Urination', 'Increased Thirst', 'Sudden Weight Loss',
	                           'Weakness', 'Increased Hunger', 'Genital Thrush', 'Visual Blurring', 'Itching',
	                           'Irritability', 'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Hair Loss',
	                           'Obesity', 'Family History']]

	t1d_Diagnosis = clean_data[['Diagnosis']]

	return t1d_features, t1d_Diagnosis


if __name__ == '__main__':
	from pathlib import Path

	file = Path.cwd().joinpath('HW2_data.csv')
	data = pd.read_csv(file, thousands=',')
	one_hot_vectors(data)
