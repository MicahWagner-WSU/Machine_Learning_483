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



steps = [
	("select", SelectColumns(["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"])),
	("scale", MinMaxScaler()),
	("cluster", KMeans(n_init = 10))
]


pipe = Pipeline(steps)

def select_type(p_type):
	tmp = data.loc[data["Type 1"] == p_type]
	return tmp[["Name", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]

def optimize_n_clusters(r,data):
	best_choice = r.start
	best_silhouette = float('-inf')
	best_cluster = []

	for n in r:
		pipe.set_params(cluster__n_clusters = n) 
		pipe.fit(data)
		poke_cluster = pipe.predict(data)
		
		scaled_data = MinMaxScaler().fit_transform(data[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]])
		score = silhouette_score(scaled_data, poke_cluster)

		print(str(n) + " clusters: " + str(score))

		if score > best_silhouette:
			best_silhouette = score
			best_choice = n
			best_cluster = poke_cluster

	print("best number of clusters: " + str(best_choice))
	print("best score: " + str(best_silhouette))
	return best_cluster

def main():

	clusters = {}
	poke_types_data = {}

	for p_types in pokemon_types:
		poke_types_data[p_types] = select_type(p_types)
		print("")
		print(p_types)
		print("-----------")
		cluster_max = len(poke_types_data[p_types])
		if cluster_max > 15:
			cluster_max = 15
		clusters[p_types] = optimize_n_clusters(range(2,cluster_max), poke_types_data[p_types])

	for p_types in pokemon_types:
		print("")
		print(p_types)
		print("-----------")

		for i in range(0, len(set(clusters[p_types]))):
			print("Cluster " + str(i))

			pokemon_cluster = poke_types_data[p_types][clusters[p_types] == i]
			print(pokemon_cluster)
			print("Mean HP: " + str(pokemon_cluster["HP"].mean()))
			print("Mean Attack: " + str(pokemon_cluster["Attack"].mean()))
			print("Mean Defense: " + str(pokemon_cluster["Defense"].mean()))
			print("Mean Sp. Atk: " + str(pokemon_cluster["Sp. Atk"].mean()))
			print("Mean Sp. Def: " + str(pokemon_cluster["Sp. Def"].mean()))
			print("Mean Speed: " + str(pokemon_cluster["Speed"].mean()))
			print("")



main()

