import numpy as np
import pandas as pd

data = pd.read_csv( 'zoo.csv' )
train = train = data.sample( frac = 0.7 )
test = data.drop( train.index )

# def main():

# def NB_probability(c, dataframe):

# def denom(c, dataframe):

# calculates the probability of a feature given a class, so P(f | c)
# if the user wants to do legs, you must specify legsX, X being the leg count
# class must be an integer
def calc_feature_given_class(f, c, dataframe):

	alpha = 1
	dimensionality = 7
	num_of_class_types = dataframe["class_type"].value_counts()
	numerator = 0

	if f[:-1] != "legs":
		numerator = ((dataframe[f] == 1) & (dataframe["class_type"] == c)).sum() + alpha
	else:
		numerator = ((dataframe[f[:-1]] == int(f[-1])) & (dataframe["class_type"] == c)).sum() + alpha

	denominator = num_of_class_types[c] + alpha * dimensionality
	return numerator / denominator




# def calc_feature_class_prob(dataframe):
# 	feature_table = {}


# 	for column_name in dataframe.columns:

# 		# avoid calculating feature prob for animal names and class type
# 		# legs need to be calculated seperately since its not a binary option
# 		if column_name not in ["animal_name", "legs", "class_type"]:
# 			# Convert values to numeric type and then calculate the probability
# 			numeric_values = pd.to_numeric(dataframe[column_name], errors='coerce')
# 			feature_table[column_name] = numeric_values.sum() / (len(dataframe) - 1)

# 		# Initialize legs columns
# 		unique_legs = dataframe["legs"].unique()
# 		for leg_number in unique_legs:
# 		    feature_table["legs" + str(leg_number)] = 0

# 		# Count the occurrences of each 'legsX' value
# 		for leg_number in dataframe["legs"]:
# 			feature_table["legs" + str(leg_number)] += 1

# 		# find the probability of having each leg type
# 		for leg_number in unique_legs:
# 			col_name = "legs" + str(leg_number)
# 			feature_table[col_name] /= (len(dataframe.index) - 1)



# 	return feature_table

print(calc_feature_given_class("eggs", 1, train))