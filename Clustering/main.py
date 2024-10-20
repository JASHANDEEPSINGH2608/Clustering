import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
import numpy as np


df = pd.read_csv('facebookdataset.csv')


X = df.values 


def compute_metrics(X, labels):
    n_clusters = len(set(labels))
    
    
    if n_clusters > 1:
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
    else:
        silhouette = np.nan  
        calinski_harabasz = np.nan
        davies_bouldin = np.nan

    return {
        "Silhouette": silhouette,
        "Calinski-Harabasz": calinski_harabasz,
        "Davies-Bouldin": davies_bouldin
    }


def get_preprocessing_pipeline(with_normalization=False, with_pca=False, with_transform=False):
    steps = []
    
    if with_normalization:
        steps.append(('scaler', StandardScaler()))
    
    if with_transform:
        steps.append(('custom_transform', StandardScaler()))  
    
    if with_pca:
        steps.append(('pca', PCA(n_components=2)))
    
    return Pipeline(steps) if steps else None


clustering_methods = {
    "KMeans": KMeans,
    "Hierarchical": AgglomerativeClustering,
    "MeanShift": MeanShift
}


preprocessing_scenarios = {
    "No Data Processing": (False, False, False),
    "Using Normalization": (True, False, False),
    "Using Transform": (False, False, True),
    "Using PCA": (False, True, False),
    "Using T+N": (True, False, True),
    "T+N+PCA": (True, True, True)
}


cluster_values = [3, 4, 5]


results = []


for clustering_method_name, clustering_method in clustering_methods.items():
    for scenario_name, (with_normalization, with_pca, with_transform) in preprocessing_scenarios.items():
        
        preprocessing_pipeline = get_preprocessing_pipeline(with_normalization, with_pca, with_transform)
        
        
        if preprocessing_pipeline:
            X_transformed = preprocessing_pipeline.fit_transform(X)
        else:
            X_transformed = X
        
        
        if clustering_method_name == "MeanShift":
            cluster_model = clustering_method()
            labels = cluster_model.fit_predict(X_transformed)
            
            
            metrics = compute_metrics(X_transformed, labels)
            metrics["Parameters"] = clustering_method_name
            metrics["Scenario"] = scenario_name
            metrics["Clusters (c)"] = "Auto"
            
           
            results.append(metrics)
        
        else:
            
            for c in cluster_values:
                cluster_model = clustering_method(n_clusters=c)
                labels = cluster_model.fit_predict(X_transformed)
                
                
                metrics = compute_metrics(X_transformed, labels)
                metrics["Parameters"] = clustering_method_name
                metrics["Scenario"] = scenario_name
                metrics["Clusters (c)"] = c
                
               
                results.append(metrics)


df_results = pd.DataFrame(results)

# Save results to an Excel file
output_file = 'clustering_results.xlsx'
df_results.to_excel(output_file, index=False)
