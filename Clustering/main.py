import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
import numpy as np


df = pd.read_csv('facebookdataset.csv')

# If necessary, preprocess your dataset (e.g., drop non-numerical columns)
X = df.values  # Convert to numpy array

# Function to compute metrics
def compute_metrics(X, labels):
    n_clusters = len(set(labels))
    
    # Ensure there are at least 2 unique clusters for silhouette score and Calinski-Harabasz
    if n_clusters > 1:
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
    else:
        silhouette = np.nan  # Assign NaN if less than 2 clusters
        calinski_harabasz = np.nan
        davies_bouldin = np.nan

    return {
        "Silhouette": silhouette,
        "Calinski-Harabasz": calinski_harabasz,
        "Davies-Bouldin": davies_bouldin
    }

# Different preprocessing pipelines
def get_preprocessing_pipeline(with_normalization=False, with_pca=False, with_transform=False):
    steps = []
    
    if with_normalization:
        steps.append(('scaler', StandardScaler()))
    
    if with_transform:
        steps.append(('custom_transform', StandardScaler()))  # Placeholder for actual transformation
    
    if with_pca:
        steps.append(('pca', PCA(n_components=2)))
    
    return Pipeline(steps) if steps else None

# Clustering methods
clustering_methods = {
    "KMeans": KMeans,
    "Hierarchical": AgglomerativeClustering,
    "MeanShift": MeanShift
}

# Preprocessing scenarios
preprocessing_scenarios = {
    "No Data Processing": (False, False, False),
    "Using Normalization": (True, False, False),
    "Using Transform": (False, False, True),
    "Using PCA": (False, True, False),
    "Using T+N": (True, False, True),
    "T+N+PCA": (True, True, True)
}

# Define number of clusters to use (only for KMeans and Hierarchical)
cluster_values = [3, 4, 5]

# Placeholder for results
results = []

# Iterate over clustering methods
for clustering_method_name, clustering_method in clustering_methods.items():
    for scenario_name, (with_normalization, with_pca, with_transform) in preprocessing_scenarios.items():
        # Preprocessing pipeline
        preprocessing_pipeline = get_preprocessing_pipeline(with_normalization, with_pca, with_transform)
        
        # Preprocess the data if a pipeline is provided
        if preprocessing_pipeline:
            X_transformed = preprocessing_pipeline.fit_transform(X)
        else:
            X_transformed = X
        
        # Special case for MeanShift (no need for n_clusters)
        if clustering_method_name == "MeanShift":
            cluster_model = clustering_method()
            labels = cluster_model.fit_predict(X_transformed)
            
            # Calculate metrics
            metrics = compute_metrics(X_transformed, labels)
            metrics["Parameters"] = clustering_method_name
            metrics["Scenario"] = scenario_name
            metrics["Clusters (c)"] = "Auto"
            
            # Append to results
            results.append(metrics)
        
        else:
            # For KMeans and Hierarchical (require n_clusters)
            for c in cluster_values:
                cluster_model = clustering_method(n_clusters=c)
                labels = cluster_model.fit_predict(X_transformed)
                
                # Calculate metrics
                metrics = compute_metrics(X_transformed, labels)
                metrics["Parameters"] = clustering_method_name
                metrics["Scenario"] = scenario_name
                metrics["Clusters (c)"] = c
                
                # Append to results
                results.append(metrics)

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Save results to an Excel file
output_file = 'clustering_results.xlsx'
df_results.to_excel(output_file, index=False)
