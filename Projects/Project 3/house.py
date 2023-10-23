from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

class SelectColumns( BaseEstimator, TransformerMixin ):

	def __init__( self, columns ):
		self.columns = columns

	def fit( self, xs, ys, **params ):
		return self

	def transform( self, xs ):
		return xs[ self.columns ].fillna(0)


grid = { 
'column_select__columns': [

		[ 
		'Gr Liv Area', 
		'Overall Qual', 
		'Year Built', 
		'Year Remod/Add',  
		'Bsmt Qual_Ex', 
		'BsmtFin Type 1_GLQ', 
		'Kitchen Qual_Ex', 
		'Garage Area',
		'Fireplaces',
		'Total Bsmt SF',
		'Lot Shape_IR1'],
	],
'linear_regression': [
	LinearRegression( n_jobs = -1 ), 
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

steps = [
    ('column_select', SelectColumns([])),  
    ('linear_regression', None),  
]

pipe = Pipeline(steps)
search = GridSearchCV(pipe, grid, scoring='r2', n_jobs=-1)

data = pd.read_csv("AmesHousing.csv")
unexpanded_xs = data.drop(columns=["SalePrice", "Neighborhood"])
xs = pd.get_dummies(unexpanded_xs, dtype=float)
ys = data["SalePrice"]


search.fit(xs, ys)


print(search.best_score_)
print("\n", search.best_params_)