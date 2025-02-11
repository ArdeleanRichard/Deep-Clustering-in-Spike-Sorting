from clustpy.deep import ENRC

from dataset_parsing import simulations_dataset as ds

X, y = ds.get_dataset_simulation(simNr=8)

model = ENRC(n_clusters=[3,3,3], embedding_size=10, random_state=42)
model.fit_predict(X)