algorithm_order = [
    #Partitional Clustering Algorithms
    # These algorithms partition the dataset into a predefined number of clusters or groups.
    "kmeans",
    "xmeans",
    "ldakmeans",
    "subkmeans",
    "specialK",
    "spectral",
    "affinity", # type?
    
    # Density-Based Clustering Algorithms
    # These algorithms identify clusters based on the density of points in the data space, often handling noise effectively.
    "meanshift",

    "dbscan",
    "optics",
    "mddbscan",
    "mdbscan",
    "amddbscan",

    "hdbscan",


    # Hierarchical Clustering Algorithms
    # These algorithms build a tree structure, merging or splitting clusters step by step.
    "agglomerative",
    "birch",
    "cure",
    "rock",
    "diana",
    
    # Grid-Based Clustering Algorithms
    "bang",
    "clique",
    
    # Statistical Clustering Evaluation
    # These methods are used to determine the optimal number of clusters or validate clustering results.
    "gapStatistic",

    # Model-Based Clustering Algorithms
    # These algorithms assume that the data is generated from a mixture of underlying probability distributions.
    "gmeans",
    "pgmeans",
    "dipMeans",

    # Subspace Clustering Algorithms
    # These algorithms discover clusters in subspaces of high-dimensional data.
    "bsas",
    "mbsas",
    "ttsas",

    # Constraint-Based Clustering Algorithms
    # These algorithms incorporate external constraints into the clustering process, such as must-link or cannot-link constraints.
    "dipInit",
    "dipNSub",
    "skinnydip",
    "projectedDipMeans",
    "syncsom",
    "somsc",


    # Deep clustering
    # These methods use deep neural networks to learn clustering representations
    "autoclustering",
    "acedec",
    "aec",
    "dcn",
    "ddc",
    "dec",
    "deepect",
    "dipdeck",
    "dipencoder",
    "dkm",
    #"enrc", ## no results on any datasets with any parameters
    "idec",
    "n2d",
    "vade",

]