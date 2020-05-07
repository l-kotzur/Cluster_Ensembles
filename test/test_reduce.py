import numpy as np
import Cluster_Ensembles as CE




def test_cluster_ensemble():
    cluster_runs = np.random.randint(0, 50, (50, 15000))
    consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose = True, N_clusters_max = 50)


if __name__ == "__main__":
    test_cluster_ensemble()