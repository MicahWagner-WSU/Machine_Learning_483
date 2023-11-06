import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("Pokemon.csv")
data.drop(columns=["Type 2", "Generation", "Legendary"])

def select_type(p_type):
	return data.loc[data["Type 1"] == p_type]

print(select_type("Grass"))