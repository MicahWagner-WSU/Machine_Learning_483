import numpy as np
import pandas as pd
import math 

data = pd.read_csv( 'zoo.csv' )
train = data.sample( frac = 0.7 )
test = data.drop( train.index )

# def main():

def calc_naive_bayes(c,row, dataframe):
	class_type = dataframe["class_type"].value_counts()
	nb = math.log(class_type[c] / class_type.sum(), 2)
	features = dataframe.drop("animal_name", axis=1).iloc[row]
	for label in features.index:
		nb += math.log(calc_feature_given_class(label,c,features[label],dataframe))
	return 2**nb

def calc_denominator(row, dataframe):
	denominator = 0
	# print(type(dataframe["class_type"].value_counts()))
	for current_class in dataframe["class_type"].value_counts().index:
		denominator += calc_naive_bayes(int(current_class), row, dataframe)
	print(denominator)
	return denominator


# calculates the probability of a feature given a class, so P(f | c)
# the user specified weather we are checking for f = true or f = false in the f_val arguemnt
# f_val = 0 to signify the feature = false, and f_val = 1 to signify the feature = true 
# if the user wants to do legs, you must specify the leg count in f_val
# class must be an integer
def calc_feature_given_class(f, c, f_val, dataframe):

	alpha = 0.01
	dimensionality = 7
	class_types = dataframe["class_type"].value_counts()
	numerator = 0

	numerator = ((dataframe[f] == f_val) & (dataframe["class_type"] == c)).sum() + alpha

	denominator = class_types[c] + alpha * dimensionality
	return numerator / denominator


print(calc_naive_bayes(1, 3, train) / calc_denominator(3, train))