import numpy as np
import pandas as pd
import math 
import sys
from functools import cache

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
		prediction = predict(test.iloc[i])
		predicted_class.append(prediction[0])
		probability.append(prediction[1])
		correct.append(prediction[2])
	test.insert(len(test.columns), 'predicted', predicted_class)
	test.insert(len(test.columns), 'probability', probability)
	test.insert(len(test.columns), 'correct?', correct)
	test.to_csv(sys.stdout, index=False)

# returns an array with the first index being the class predicted,
# the second being the probability of the prediction,
# and the last being whether the guess was correct or not
def predict(predict_x):
	probability_array = []
	class_array = [] 
	class_types = [1,2,3,4,5,6,7]

	# loop over each class to see the heighest probability 
	for c in class_types:
		num = calc_naive_bayes(c, predict_x)
		den = 0

		#convert the naive bayse calculations into probabilities
		for current_class in class_types:
			den += calc_naive_bayes(current_class, predict_x)
		probability_array.append(num / den)
		class_array.append(c)

	highest_prob_index = probability_array.index(max(probability_array))
	class_predicted = class_array[highest_prob_index]
	probability =  probability_array[highest_prob_index]
	return [class_predicted, probability, "CORRECT" if class_predicted == predict_x['class_type'] else "WRONG"]

# calculates naive bayes given a class and a set of features
# this function does not calculate the probability, just p(c)*p(f1|c)*p(f2|c)...
def calc_naive_bayes(c,row):
	class_count = (train["class_type"] == c).sum()
	nb = math.log((class_count + alpha) / (train["class_type"].value_counts().sum() + alpha * dimensionality), 2)

	# exclude animal name and class type
	for feature in row[1:-1].keys():
		nb += math.log(calc_feature_given_class(feature,c,row[feature]),2)
	return 2**nb


# calculates the probability of a feature given a class, so P(f | c)
# the user specifies weather we are checking for f = true or f = false in the f_val arguemnt
# f_val = 0 to signify the feature = false, and f_val = 1 to signify the feature = true 
# if the user wants to do legs, you must specify the leg count in f_val
# class must be an integer
@cache
def calc_feature_given_class(f, c, f_val):
	class_types = train["class_type"].value_counts()
	numerator = ((train[f] == f_val) & (train["class_type"] == c)).sum() + alpha

	# the .get will ensure that if c doesn't exist, it returns 0 instead of a keyerror
	denominator = class_types.get(c, default = 0) + alpha * dimensionality
	return numerator / denominator

main()