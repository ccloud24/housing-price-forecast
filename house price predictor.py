import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv("house_prices.csv")

# Step 2: Select features (X) and target variable (y)
X = df[['sqft', 'bedrooms', 'bathrooms', 'location']]  # Features
y = df['price']  # Target variable

# Step 3: Convert categorical data (location) to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['location'])

# Step 4: Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Normalize numerical features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Define the neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer
    keras.layers.Dense(32, activation='relu'),  # Second hidden layer
    keras.layers.Dense(1)  # Output layer (predicting a continuous value)
])

# Step 7: Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Step 8: Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Step 9: Evaluate the model on test data
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Mean Absolute Error: ${test_mae:.2f}')

# Step 10: Make a prediction for a new house

# Example new house data (needs to have the same columns as X_train)
sample_house = pd.DataFrame({
    'sqft': [5000],  # Example input: 5000 sqft
    'bedrooms': [5],  # 5 bedrooms
    'bathrooms': [6],  # 6 bathrooms
    'location': ['98001']  # Assume location is one of the valid categories from training data
})

# One-hot encode the location feature of the new data to match the training data's one-hot encoding
sample_house = pd.get_dummies(sample_house, columns=['location'])

# Align the columns to match X_train
sample_house = sample_house.reindex(columns=X.columns, fill_value=0)

# Scale the features using the same scaler fitted to X_train
sample_house_scaled = scaler.transform(sample_house)

# Predict the price
predicted_price = model.predict(sample_house_scaled)
print(f'Predicted Price: ${predicted_price[0][0]:,.2f}')