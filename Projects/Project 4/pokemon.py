import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

data = pd.read_csv("Pokemon.csv")
data.drop(columns=["Type 2", "Generation", "Legendary"])
pokemon_types = data["Type 1"].unique()
print(pokemon_types)


steps = [
	("scale", MinMaxScaler()),
	("cluster", KMeans())
]


pipe = Pipeline(steps)

def select_type(p_type):
	return data.loc[data["Type 1"] == p_type]

# make function that finds best cluster value, refer quiz answers from slide 12
# loop over each poke type and call the above function, print results in specifc way