import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('/content/Computer.csv')

# Select input and output
X = df[['Units']]
y = df['Minutes']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Display results
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# Plot graph
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.title("Units vs Minutes")
plt.xlabel("Units")
plt.ylabel("Minutes")
plt.show()

# Predict minutes for 11 units
print("Predicted Minutes:", model.predict([[11]])[0])
