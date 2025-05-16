import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import Row
import warnings
import plotly.graph_objects as go
from pyspark.sql.functions import col

warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value 1 overridden")

# Initialize Spark session
spark = SparkSession.builder.appName('HW4_6D_Data_Hunt').getOrCreate()

# Load data as RDD
file_path = 'space.dat'
rdd = spark.sparkContext.textFile(file_path)

# Transform RDD data to rows with schema for Spark DataFrame
data_rdd = rdd.map(lambda line: Row(*map(float, line.split(","))))
columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
df = spark.createDataFrame(data_rdd, schema=columns)

# Assemble features into a vector column
assembler = VectorAssembler(inputCols=columns, outputCol='features')
df_with_features = assembler.transform(df).cache()

# Determine optimal k using silhouette score
k_range = range(2, 11)
silhouette_scores_kmeans = []
evaluator = ClusteringEvaluator()

for k in k_range:
    kmeans = KMeans(k=k, seed=1, featuresCol='features', predictionCol='prediction')
    kmeans_model = kmeans.fit(df_with_features)
    kmeans_labels = kmeans_model.transform(df_with_features)
    score = evaluator.evaluate(kmeans_labels)
    silhouette_scores_kmeans.append(score)

# Choose the best k
best_k = k_range[silhouette_scores_kmeans.index(max(silhouette_scores_kmeans))]
print(f"Optimal k for KMeans: {best_k}")
print(f"Silhouette Score : {max(silhouette_scores_kmeans)}")

# Run KMeans with the optimal k
kmeans = KMeans(k=best_k, seed=1, featuresCol='features', predictionCol='prediction')
kmeans_model = kmeans.fit(df_with_features)
kmeans_labels = kmeans_model.transform(df_with_features)

# Convert to Pandas
df_pandas = df.toPandas()
df_pandas['cluster'] = kmeans_labels.select('prediction').toPandas()

# Get 6D cluster centers
cluster_centers = kmeans_model.clusterCenters()

# Print cluster shape, center, and number of points information
for i in range(best_k):
    cluster_points = kmeans_labels.filter(col("prediction") == i)
    count = cluster_points.count()
    
    # Convert filtered cluster points to pandas to calculate sizes in each dimension
    cluster_points_pandas = cluster_points.select(columns).toPandas()
    
    # Calculate size in each dimension (min, max, and range)
    min_values = cluster_points_pandas.min(axis=0)
    max_values = cluster_points_pandas.max(axis=0)
    size_in_each_dimension = max_values - min_values  # This is the range in each dimension
    
    # 6D Shape (Bounding Box) - the 6D "shape" is defined by the min and max in each dimension
    bounding_box = list(zip(min_values, max_values))
    
    # Print cluster shape information
    print(f"Cluster {i}:")
    print(f" - Center: {cluster_centers[i]}")
    print(f" - Count: {count}")
    print(f" - Size in Each Dimension: {size_in_each_dimension}")
    print(f" - Bounding Box: {bounding_box}\n")

# Plot each cluster separately in 3D PCA plot
for i in range(best_k):
    # Filter data for the current cluster
    cluster_data = df_pandas[df_pandas['cluster'] == i][columns].values
    
    # Convert to Spark DataFrame for PCA
    cluster_spark_df = spark.createDataFrame(pd.DataFrame(cluster_data, columns=columns))
    assembler = VectorAssembler(inputCols=columns, outputCol='features')
    cluster_with_features = assembler.transform(cluster_spark_df)
    
    # Apply PCA to reduce to 6 dimensions
    pca = PCA(k=6, inputCol='features', outputCol='pca_features')
    pca_model = pca.fit(cluster_with_features)
    pca_result = pca_model.transform(cluster_with_features)
    
    # Convert PCA results to Pandas for plotting
    pca_pd = pca_result.select('pca_features').toPandas()
    pca_coordinates = np.array([x.toArray() for x in pca_pd['pca_features']])
    
    # Explained variance ratios for all 6 PCs
    explained_variance_ratios = pca_model.explainedVariance.toArray()
    explained_variance_text = (
        f"Explained Variance:\n"
        f"PCA 1: {explained_variance_ratios[0]:.2%}\n"
        f"PCA 2: {explained_variance_ratios[1]:.2%}\n"
        f"PCA 3: {explained_variance_ratios[2]:.2%}\n"
        f"PCA 4: {explained_variance_ratios[3]:.2%}\n"
        f"PCA 5: {explained_variance_ratios[4]:.2%}\n"
        f"PCA 6: {explained_variance_ratios[5]:.2%}"
     )
    
    # Create a 3D plot for the cluster
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_coordinates[:, 0], pca_coordinates[:, 1], pca_coordinates[:, 2], c='b', alpha=0.6)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title(f'Cluster {i}')
    ax.text2D(0.05, 0.95, explained_variance_text, transform=ax.transAxes, fontsize=8, verticalalignment='top')
    
    # Save each plot separately
    plt.savefig(f'cluster_{i}_pca_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create interactive 3D plot using Plotly
    fig = go.Figure(data=[go.Scatter3d(
    x=pca_coordinates[:, 0],
    y=pca_coordinates[:, 1],
    z=pca_coordinates[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
        opacity=0.6
        )
    )])
    
    # Add axis labels and title
    fig.update_layout(
    scene=dict(
        xaxis_title='PCA 1',
        yaxis_title='PCA 2',
        zaxis_title='PCA 3'
        ),
    title=f'Cluster {i} - 3D PCA Plot',
    annotations=[
        dict(
            x=0.5,
            y=0.95,
            text=explained_variance_text,
            showarrow=False,
            font=dict(size=10),
            xref="paper", yref="paper"
            )
        ]
    )
    
# Save the interactive plot as an HTML file
    plot_filename = f'cluster_{i}_pca_3d_interactive.html'
    fig.write_html(plot_filename)
    print(f"Saved interactive plot for Cluster {i} to {plot_filename}")

# Stop Spark session
spark.stop()
