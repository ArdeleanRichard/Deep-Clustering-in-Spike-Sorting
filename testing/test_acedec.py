import numpy as np
from clustpy.deep import ACeDeC
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score, calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix

from dataset_parsing import simulations_dataset as ds
simulation_number = 4
X, y_true = ds.get_dataset_simulation(simNr=simulation_number)
print(X.shape)

model = ACeDeC(n_clusters=len(np.unique(y_true)))
y_pred = model.fit_predict(X)

ari = adjusted_rand_score(y_true, y_pred)
ami = adjusted_mutual_info_score(y_true, y_pred)
contingency_mat = contingency_matrix(y_true, y_pred)
purity = np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)
ss = silhouette_score(X, y_pred)
chs = calinski_harabasz_score(X, y_pred)
dbs = davies_bouldin_score(X, y_pred)

print(ari, ami, purity, ss, chs, dbs)