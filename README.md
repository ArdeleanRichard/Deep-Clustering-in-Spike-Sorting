# A Study of Deep Clustering in Spike Sorting

**Benchmarking Deep Clustering Algorithms for Next-Generation Spike Sorting**

[![DOI](https://img.shields.io/badge/DOI-10.1007/s12021--025--09751--4-blue)](https://doi.org/10.1007/s12021-025-09751-4)

This repository contains the code and configurations used in the study:

**"A Study of Deep Clustering in Spike Sorting"**
by **Eugen-Richard Ardelean** and **Raluca Laura Portase**, published in *Neuroinformatics*, 2025.

---

## Overview

This project provides a large-scale **benchmark of 12 deep clustering algorithms** against the traditional spike sorting pipeline (feature extraction combined with K-means clustering).

**Traditional spike sorting** separates representation learning and clustering into distinct steps, which may not optimally capture the complex structure of spike data. **Deep clustering** addresses this by performing a dual optimization, effectively learning non-linear representations tailored for clustering. Our findings indicate that these deep clustering approaches are the most suitable methods for accurately identifying individual neuronal activity in modern multi-electrode recordings.

---

## Datasets

### Synthetic Spike Waveforms (Pedreira et al., 2012)
- **Description**: 95 single-channel synthetic datasets derived from real monkey recordings.  
- **Characteristics**: 2–20 clusters per dataset, ~9,300 spikes on average, including multi-unit clusters.  
- **Usage**: Benchmarking feature extraction across diverse conditions.  
- **Access**: Publicly available.  

### Real Datasets (spe-1, Marques-Smith et al., 2018/2020)
- **Description**: Patch-clamp + 384-channel CMOS extracellular recordings in rat cortex.  
- **Ground Truth**: Dual intracellular/extracellular data for 21 neurons.  
- **Datasets Used**: c28 and c37.  
---

## Methods Benchmarked

The study compares deep clustering against the traditional two-stage pipeline.

### Deep Clustering Algorithms
A total of **12 deep clustering algorithms** were benchmarked.
- **Key Algorithms**: **ACeDeC, AEC, DCN, DDC, DEC, DeepECT, DipDECK, DipEncoder, DKM, IDEC, VaDE, N2D**.

### Traditional Pipeline
- **Feature Extraction**: Linear (**PCA, ICA**) and Non-linear/Manifold (**Isomap, LLE, t-SNE, Diffusion Map**) methods.
- **Clustering**: Clustering was performed using **K-Means** for evaluation of the feature extraction methods.

---

## Results Summary

Performance was evaluated using six clustering metrics: **Adjusted Rand Index (ARI), Adjusted Mutual Information (AMI), Purity, Silhouette Score (SS), Calinski-Harabasz Score (CHS), and Davies-Bouldin Score (DBS)**.

**Key findings:**
- **Superior Performance**: A subset of deep clustering algorithms—particularly **ACeDeC, DDC, DEC, IDEC, and VaDE**—significantly outperformed traditional methods, especially as dataset complexity increased.
- **Top Performers**:
    - **DDC** excelled on datasets with low to medium cluster counts.
    - **DEC, IDEC, and VaDE** were the top performers for datasets with a medium to high number of clusters.

---

## Citation

If you use this work, please cite:

```bibtex
@article{Ardelean2025DeepClustering,
  title     = {A Study of Deep Clustering in Spike Sorting},
  author    = {Ardelean, Eugen-Richard and Portase, Raluca Laura},
  journal   = {Neuroinformatics},
  year      = {2025},
  volume    = {23},
  pages     = {51},
  doi       = {10.1007/s12021-025-09751-4},
}
```

---

## 📬 Contact

For questions, please contact:
📧 [ardeleaneugenrichard@gmail.com](mailto:ardeleaneugenrichard@gmail.com)

---
