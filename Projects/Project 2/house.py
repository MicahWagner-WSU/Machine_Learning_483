from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

class SelectColumns( BaseEstimator, TransformerMixin ):
	# pass the function we want to apply to the column 'SalePrice’
	def __init__( self, columns ):
		self.columns = columns
	# don't need to do anything
	def fit( self, xs, ys, **params ):
		return self
	# actually perform the selection
	def transform( self, xs ):
		return xs[ self.columns ].fillna(0)


grid = { 
'column_select__columns': [
		[ 'Gr Liv Area' ],
		[ 'Gr Liv Area', 'Overall Qual' ],
		[ 'Gr Liv Area', 'Overall Qual', 'Year Built', 'Year Remod/Add'],
		[ 'Gr Liv Area', 'Overall Qual', 'Year Built', 'Year Remod/Add', 'Neighborhood_NridgHt'],
		[ 'Gr Liv Area', 'Overall Qual', 'Year Built', 'Year Remod/Add', 'Neighborhood_NridgHt', 'Foundation_PConc'],
		[ 'Gr Liv Area', 'Overall Qual', 'Year Built', 'Year Remod/Add', 'Neighborhood_NridgHt', 'Foundation_PConc', 'Bsmt Qual_Ex'],
		[ 'Gr Liv Area', 'Overall Qual', 'Year Built', 'Year Remod/Add', 'Neighborhood_NridgHt', 'Foundation_PConc', 'Bsmt Qual_Ex', 'BsmtFin Type 1_GLQ'],
		[ 'Gr Liv Area', 'Overall Qual', 'Year Built', 'Year Remod/Add', 'Neighborhood_NridgHt', 'Foundation_PConc', 'Bsmt Qual_Ex', 'BsmtFin Type 1_GLQ', 'Full Bath'],
	],
'linear_regression': [
	LinearRegression( n_jobs = -1 ), # no transformation
	TransformedTargetRegressor(
		LinearRegression( n_jobs = -1 ),
		func = np.sqrt,
		inverse_func = np.square ),
	TransformedTargetRegressor(
		LinearRegression( n_jobs = -1 ),
		func = np.cbrt,
		inverse_func = lambda y: np.power( y, 3 ) ),
	TransformedTargetRegressor(
		LinearRegression( n_jobs = -1 ),
		func = np.log,
		inverse_func = np.exp),
	]
}

# Create a pipeline with column selection and regression
steps = [
    ('column_select', SelectColumns([])),  
    ('linear_regression', None),  
]

pipe = Pipeline(steps)

# Create a grid search using the pipeline
search = GridSearchCV(pipe, grid, scoring='r2', n_jobs=-1)

# Load the data
data = pd.read_csv("AmesHousing.csv")
tmp_xs = data.drop(columns=["SalePrice"])
xs = pd.get_dummies(tmp_xs, dtype=float)
ys = data["SalePrice"]

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(xs, ys, train_size=0.7)

# Fit the pipeline using the training data
search.fit(xs, ys)

# Print the best score and make predictions
print("Best R^2 Score:", search.best_score_)
print("Best params:", search.best_params_)

print("Predictions for the first three samples:")

print(search.predict(xs.iloc[:3]))











