import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# Load dataset
data = pd.read_csv("data.csv")

# Remove unnecessary columns
data = data.drop(["id", "Unnamed: 32"], axis=1)

# Convert diagnosis into numbers
data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})


# Separate input and target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Build ANN
model = Sequential()

model.add(Dense(64, activation="relu", input_shape=(30,)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Train model
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)


# Make predictions
y_probability = model.predict(X_test)
y_pred = (y_probability >= 0.5).astype(int)


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Save model
model.save("breast_cancer_ann.keras")

print("\nANN model saved successfully!")