import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('/content/student_preprocessing.csv')
print("Original Data:\n", df)

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Marks'] = df['Marks'].fillna(df['Marks'].mean())

# Encode categorical data
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Scale numerical data
scaler = StandardScaler()
df[['Age', 'Marks']] = scaler.fit_transform(df[['Age', 'Marks']])

# Split dataset
train, test = train_test_split(
    df, test_size=0.2, random_state=42
)

# Display results
print("\nPreprocessed Data:\n", df)
print("\nTraining Data:\n", train)
print("\nTesting Data:\n", test)
