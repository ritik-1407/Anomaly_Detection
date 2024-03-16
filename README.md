# Anomaly-Detection-in-Time-Series-Data

This repository contains the code and datasets used for our research on anomaly detection in time series data. We have implemented and compared five different algorithms:

1. K-Nearest Neighbors (KNN)
2. Long Short-Term Memory Autoencoder (LSTM-AE)
3. Isolation Forest
4. Deep Ant
5. LoOp

We have evaluated the performance of these algorithms on both multivariate and univariate time series datasets.

## Datasets

### Multivariate Dataset (SWAT)

For the multivariate analysis, we used the SWAT dataset, which contains approximately 4.5 lakhs (450,000) records. This dataset is significant in size and complexity, making it suitable for testing the robustness of our anomaly detection algorithms in real-world scenarios.

### Univariate Dataset (Synthetic Data)

For the univariate analysis, we generated synthetic time series data using the GUENTAG algorithm. This allowed us to control the characteristics of the data, making it ideal for studying the behavior of our algorithms under controlled conditions. Users can customize the synthetic data by providing input parameters to the GUENTAG algorithm to generate specific types of time series data.

## Repository Structure

- `src/`: This directory contains the source code for implementing and evaluating the anomaly detection algorithms.
- `datasets/`: Here, you can find the datasets used in our experiments, including the SWAT dataset and the synthetic data generated using GUENTAG.
- `plots/`: This folder stores the results of our experiments, including evaluation metrics, visualizations, and any other relevant outputs.
- `docs/`: If you have any documentation related to your research or code, you can place it here.
- `LICENSE`: The license file for this repository.

## Getting Started

To replicate our experiments or use our code, please refer to the documentation within the `src/` directory for instructions on setting up your environment and running the algorithms. Make sure to install any required dependencies before proceeding.

## Citation

If you find our work useful in your research, please consider citing our paper (if available) or this repository. You can find citation details in the respective research paper or in the repository's `CITATION.md` file.

## Contributors

From:  
Aishwarya Bodkhe,Avishek Pathania,Prajna Shetty,Sanchita Singh,Ritik Gupta


## License
