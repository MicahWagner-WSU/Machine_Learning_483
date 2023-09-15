import numpy as np
import pandas as pd
import math 
import sys

data = pd.read_csv( 'zoo.csv' )
train = data.sample( frac = 0.7 )
test = data.drop( train.index )

alpha = 0.01
dimensionality = 7

def main():
	predicted_class = []
	probability = []
	correct = []

	for i in range(0, len(test.index)):
		prediction = predict(i, test)
		predicted_class.append(prediction[1])
		probability.append(prediction[0])
		correct.append(prediction[2])
	test["predicted"] = predicted_class
	test["probability"] = probability
	test["correct?"] = correct
	test.to_csv(sys.stdout, index=False)
	
	
def predict(row, dataframe):
	class_types = data["class_type"].unique()
	actual_class_type = dataframe.iloc[row].loc["class_type"]
	highest_prediction = 0
	class_type = -1
	correct = "?"
	for i in class_types:
		prediction = calc_naive_bayes(i, row, dataframe) / calc_denominator(row, dataframe)
		if (prediction > highest_prediction):
			highest_prediction = prediction
			class_type = i
		else:
			continue
	if(class_type == actual_class_type):
		correct = "CORRECT"
	else:
		correct = "WRONG"
	return [highest_prediction, class_type, correct]

def calc_naive_bayes(c,row, dataframe):
	class_type = dataframe["class_type"].value_counts()
	class_count = (train["class_type"] == c).sum()
	nb = math.log((class_count + alpha)/ (train["class_type"].value_counts().sum() + alpha * dimensionality), 2)
	features = dataframe.drop("animal_name", axis=1).iloc[row]
	for label in features.index:
		nb += math.log(calc_feature_given_class(label,int(c),features[label]),2)
	return 2**nb

def calc_denominator(row, dataframe):
	denominator = 0
	class_types = data["class_type"].unique()
	for current_class in class_types:
		denominator += calc_naive_bayes(current_class, row, dataframe)
	return denominator


# calculates the probability of a feature given a class, so P(f | c)
# the user specified weather we are checking for f = true or f = false in the f_val arguemnt
# f_val = 0 to signify the feature = false, and f_val = 1 to signify the feature = true 
# if the user wants to do legs, you must specify the leg count in f_val
# class must be an integer
def calc_feature_given_class(f, c, f_val):
	class_types = train["class_type"].value_counts()
	numerator = 0

	numerator = ((train[f] == f_val) & (train["class_type"] == c)).sum() + alpha

	denominator = class_types[c] + alpha * dimensionality
	return numerator / denominator

main()