import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Classification using ANN")
st.write(
    "This application uses an Artificial Neural Network "
    "to classify breast cancer tumors as Benign or Malignant."
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    data = pd.read_csv("data.csv")

    data = data.drop(
        ["id", "Unnamed: 32"],
        axis=1
    )

    data["diagnosis"] = data["diagnosis"].map({
        "B": 0,
        "M": 1
    })

    return data


data = load_data()


# --------------------------------------------------
# SEPARATE INPUT AND TARGET
# --------------------------------------------------

X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

@st.cache_resource
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build ANN
    model = Sequential()

    model.add(
        Dense(
            64,
            activation="relu",
            input_shape=(30,)
        )
    )

    model.add(
        Dense(
            32,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    # Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # Train
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    # Test prediction
    y_probability = model.predict(
        X_test_scaled,
        verbose=0
    )

    y_pred = (
        y_probability >= 0.5
    ).astype(int).ravel()

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    return (
        model,
        scaler,
        accuracy,
        X_test,
        y_test,
        y_pred,
        history
    )


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

with st.spinner("Training ANN model..."):

    (
        model,
        scaler,
        accuracy,
        X_test,
        y_test,
        y_pred,
        history
    ) = train_model(X, y)


# --------------------------------------------------
# DISPLAY ACCURACY
# --------------------------------------------------

st.success(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.subheader("🧠 ANN Architecture")

st.write("""
**Input Layer:** 30 features

**Hidden Layer 1:** 64 neurons + ReLU

**Hidden Layer 2:** 32 neurons + ReLU

**Output Layer:** 1 neuron + Sigmoid

**Optimizer:** Adam

**Loss Function:** Binary Crossentropy
""")


# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------

st.subheader("🔍 Test a Sample")

st.write(
    "Select a sample from the test dataset and let the ANN "
    "predict whether the tumor is benign or malignant."
)


# Select sample
sample_number = st.number_input(
    "Test sample number",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1
)


if st.button("Predict"):

    # Get selected sample
    sample = X_test.iloc[[sample_number]]

    # Scale sample
    sample_scaled = scaler.transform(sample)

    # Prediction
    probability = model.predict(
        sample_scaled,
        verbose=0
    )[0][0]

    prediction = (
        1 if probability >= 0.5 else 0
    )

    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"🔴 Malignant\n\n"
            f"Probability: {probability * 100:.2f}%"
        )

    else:

        st.success(
            f"🟢 Benign\n\n"
            f"Probability of Malignant: "
            f"{probability * 100:.2f}%"
        )


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

st.write(cm)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)
