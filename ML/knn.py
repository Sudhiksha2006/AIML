import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('/content/knn_dataset.csv')

# Separate features and target
X = df[['Height', 'Weight']]
y = df['Class']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict output
y_pred = model.predict(X_test)

# Display results
print("Actual values:", y_test.values)
print("Predicted values:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot graph
plt.scatter(df['Height'], df['Weight'],
            c=pd.factorize(df['Class'])[0], cmap='rainbow')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('KNN Classification')
plt.show()