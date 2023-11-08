import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
 

class SelectColumns( BaseEstimator, TransformerMixin ):

	def __init__( self, columns ):
		self.columns = columns

	def fit( self, xs, **params ):
		return self

	def transform( self, xs ):
		return xs[ self.columns ].fillna(0)

data = pd.read_csv("Pokemon.csv")
data.drop(columns=["Type 2", "Generation", "Legendary"])
pokemon_types = data["Type 1"].unique()
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = MinMaxScaler().fit_transform(data[numeric_cols])
print(pokemon_types)


steps = [
	("select", SelectColumns(["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"])),
	("cluster", KMeans(n_init = 2))
]


pipe = Pipeline(steps)

def select_type(p_type):
	return data.loc[data["Type 1"] == p_type]

def optimize_n_clusters(r,data):
	best_choice = r.start
	best_silhouette = float('-inf')

	for n in r:
		pipe.set_params(cluster__n_clusters = n) 
		pipe.fit(data)
		poke_cluster = pipe.predict(data)

		score = silhouette_score(data[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]], poke_cluster)

		print(str(n) + " clusters: " + str(score))

		if score > best_silhouette:
			best_silhouette = score
			best_choice = n

	print("best number of clusters: " + str(best_choice))
	print("best score: " + str(best_silhouette))
	return best_choice

def main():

	for p_types in pokemon_types:
		pokemon_data = select_type(p_types)
		print("")
		print(p_types)
		print("-------------")
		print(len(pokemon_data));
		optimize_n_clusters(range(2,len(pokemon_data)), pokemon_data)

main()

# make function that finds best cluster value, refer quiz answers from slide 12
# loop over each poke type and call the above function, print results in specifc way