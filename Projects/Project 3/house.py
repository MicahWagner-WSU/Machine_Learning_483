from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd

class SelectColumns( BaseEstimator, TransformerMixin ):

	def __init__( self, columns ):
		self.columns = columns

	def fit( self, xs, ys, **params ):
		return self

	def transform( self, xs ):
		return xs[ self.columns ].fillna(0)


data = pd.read_csv("AmesHousing.csv")
unexpanded_xs = data.drop(columns=["SalePrice", "Neighborhood"])
expanded_xs = pd.get_dummies(unexpanded_xs, dtype=float)
xs = expanded_xs.select_dtypes(include='number')
ys = data["SalePrice"]


grid_gradient = { 
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
'regression': [
		GradientBoostingRegressor()
	],
'regression__max_depth': [3,4,5,6,7,8,9,10],
'regression__max_features': ["sqrt", "log2"],
'regression__learning_rate': [0.1, 0.4, 0.8, 1],

}


grid_forest = { 
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
'regression': [
		RandomForestRegressor(n_jobs = -1)
	],
'regression__max_depth': [3,4,5,6,7,8,9,10],
'regression__max_features': ["sqrt", "log2"],
}

grid_tree = { 
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
'regression': [
		DecisionTreeRegressor()
	],
'regression__max_depth': [3,4,5,6,7,8,9,10],
'regression__max_features': [2,4,6,8,10,11]
}

grid_linear = { 
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
'regression': [
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
    ('regression', None),  
]
pipe = Pipeline(steps)
search_linear = GridSearchCV(pipe, grid_linear, scoring='r2', n_jobs=-1)
search_tree = GridSearchCV(pipe, grid_tree, scoring='r2', n_jobs=-1)	
search_forest = GridSearchCV(pipe, grid_forest, scoring='r2', n_jobs=-1)	
search_gradient= GridSearchCV(pipe, grid_gradient, scoring='r2', n_jobs=-1)	


search_linear.fit(xs, ys)

print("Linear Regression: ")
print("R-squared:" + str(search_linear.best_score_))
print("Best Params:" + str(search_linear.best_params_))
print("")


search_gradient.fit(xs, ys)

print("Gradient boosting:  ")
print("R-squared:" + str(search_gradient.best_score_))
print("Best Params:" + str(search_gradient.best_params_))
print("")

search_forest.fit(xs, ys)

print("Random Forest: ")
print("R-squared:" + str(search_forest.best_score_))
print("Best Params:" + str(search_forest.best_params_))
print("")


search_tree.fit(xs, ys)

print("Decision Tree: ")
print("R-squared: " + str(search_tree.best_score_))
print("Best Params: " + str(search_tree.best_params_))
print("")

