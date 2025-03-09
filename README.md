# Customer-Segmentation

# Customer Segmentation using PCA and K-Means

**Author:** SIYABULELA MATHE

## Overview

This project focuses on performing customer segmentation using Principal Component Analysis (PCA) and K-Means clustering. The goal is to understand customer behaviors based on their demographic and spending patterns, which can then be used by the marketing team to develop targeted strategies.

## Dataset

The dataset used in this project is the "Mall Customer Segmentation" dataset, available on Kaggle:

The dataset contains the following attributes:

* `CustomerID`: Unique identifier for each customer.
* `Gender`: Customer's gender.
* `Age`: Customer's age.
* `Annual Income (k$)`: Customer's annual income in thousands of dollars.
* `Spending Score (1-100)`: A score assigned to the customer based on their spending behavior.

## Workflow

The project follows these steps:

1.  **Loading Data:**
    * The dataset is loaded using pandas.
2.  **Data Exploration:**
    * Exploratory data analysis (EDA) is performed to understand the distribution of features and relationships between them using matplotlib and seaborn.
3.  **Train/Test Split:**
    * The dataset is split into training and testing sets to evaluate the model's performance.
4.  **Data Preparation:**
    * Preprocessing steps include:
        * `LabelEncoder`: To convert the 'Gender' categorical feature into numerical values.
        * `StandardScaler`: To scale the numerical features.
        * Principal Component Analysis (PCA): To reduce the dimensionality of the data.
5.  **Principal Component Analysis (PCA):**
    * PCA is applied to reduce the dimensionality of the data while retaining most of the variance.
6.  **K-Means Clustering:**
    * K-Means clustering is used to group customers into distinct segments.
    * The elbow method is used to determine the optimal number of clusters.
7.  **Pipeline:**
    * A pipeline is created to streamline the preprocessing and clustering steps.
8.  **Validation with Test Data:**
    * The trained model is used to predict clusters for the test data, and the results are analyzed.
9.  **Cluster Analysis:**
    * A heatmap is created to visualize the characteristics of each cluster.

## Imports

The following Python libraries are used:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
