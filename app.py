import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    train_df = pd.read_csv("Training.csv")
    test_df = pd.read_csv("Testing.csv")

    # Remove unwanted 'Unnamed' columns
    train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
    test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

    return train_df, test_df

train_df, test_df = load_data()

# ----------------------------
# Prepare Features & Labels
# ----------------------------
X = train_df.drop("prognosis", axis=1)
y = train_df["prognosis"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Train Model
# ----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Evaluate Model
# ----------------------------
y_pred_val = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🩺 Disease Detection App by jzmtx")
st.write(f"Model Accuracy: **{accuracy*100:.2f}%**")

st.header("Enter Your Symptoms")

symptom_list = X.columns.tolist()

# Multi-select symptoms
selected_symptoms = st.multiselect("Select the symptoms you have:", symptom_list)

# Create input vector
input_data = [0] * len(symptom_list)
for symptom in selected_symptoms:
    if symptom in symptom_list:
        input_data[symptom_list.index(symptom)] = 1

if st.button("Predict"):
    prediction = model.predict([input_data])[0]
    st.success(f"Predicted Disease: **{prediction}**")

# ----------------------------
# Test Data Prediction Preview
# ----------------------------
if st.checkbox("Show test predictions"):
    X_test = test_df.drop("prognosis", axis=1)
    y_test = test_df["prognosis"]

    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    st.write(f"Test Accuracy: **{test_acc*100:.2f}%**")
    st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": test_pred}))
