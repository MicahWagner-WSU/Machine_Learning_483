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
		return xs[ self.columns ]

grid = { 
'column_select__columns': [
		[ 'Gr Liv Area' ],
		[ 'Gr Liv Area', 'Overall Qual' ],
		[ 'Gr Liv Area', 'Overall Qual', 'Year Built' ],
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
regressor = TransformedTargetRegressor(
	LinearRegression( n_jobs = -1 ),
	func = np.sqrt,
	inverse_func = np.square
)

# Create a pipeline with column selection and regression
steps = [
    ('column_select', SelectColumns(columns=['Gr Liv Area', 'Overall Qual', 'Year Built'])),  # Adjust the columns as needed
    ('linear_regression', regressor),  # You can use any of the regressors from your 'grid'
]

pipe = Pipeline(steps)

# Create a grid search using the pipeline
search = GridSearchCV(pipe, grid, scoring='r2', n_jobs=-1)

# Load the data
data = pd.read_csv("AmesHousing.csv")
xs = data.drop(columns=["SalePrice"])
ys = data["SalePrice"]

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(xs, ys, train_size=0.7)

# Fit the pipeline using the training data
search.fit(train_x, train_y)

# Print the best score and make predictions
print("Best R^2 Score:", search.best_score_)
print("Predictions for the first three samples:")
print(search.predict(xs.iloc[:3]))











