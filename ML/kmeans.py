import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('/content/student_performance.csv')

# Select numerical columns
X = df.select_dtypes(include='number')

# Handle missing values
X = X.fillna(X.mean())

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Display results
print(df)

# Plot clusters using first two features
plt.scatter(X.iloc[:, 0], X.iloc[:, 1],
            c=df['Cluster'], cmap='rainbow')

# Convert centroids to original scale
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Plot centroids
plt.scatter(centers[:, 0], centers[:, 1],
            color='black', marker='X', s=100,
            label='Centroids')

plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title('K-Means Clustering')
plt.legend()
plt.show()