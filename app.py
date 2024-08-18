import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Define the variables and their dropdown options with mappings
variables = {
    "age": {"type": "number"},
    "sex": {"type": "select", "options": ["female", "male"]},
    "cp": {"type": "select", "options": ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"]},
    "trestbps": {"type": "number"},
    "chol": {"type": "number"},
    "fbs": {"type": "select", "options": ["lower than 120mg/ml", "greater than 120mg/ml"]},
    "restecg": {"type": "select", "options": ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"]},
    "thalach": {"type": "number"},
    "exang": {"type": "select", "options": ["no", "yes"]},
    "oldpeak": {"type": "number"},
    "slope": {"type": "select", "options": ["upsloping", "flat", "downsloping"]},
    "ca": {"type": "number"},
    "thal": {"type": "select", "options": ["normal", "fixed defect", "reversable defect"]}
}

# Mappings to convert string values to the numeric values expected by the model
mappings = {
    "sex": {"female": 0, "male": 1},
    "cp": {"typical angina": 1, "atypical angina": 2, "non-anginal pain": 3, "asymptomatic": 4},
    "fbs": {"lower than 120mg/ml": 0, "greater than 120mg/ml": 1},
    "restecg": {"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2},
    "exang": {"no": 0, "yes": 1},
    "slope": {"upsloping": 1, "flat": 2, "downsloping": 3},
    "thal": {"normal": 1, "fixed defect": 2, "reversable defect": 3}
}

st.title('Heart Disease Prediction')
st.write("The prediction can either be `heart disease` or `No heart disease` ")

# Create input fields
data = []
for var, info in variables.items():
    if info["type"] == "select":
        value = st.selectbox(var.capitalize(), info["options"])
        # Convert string input to numeric value using mappings
        data.append(mappings[var][value])
    else:
        value = st.number_input(var.capitalize(), step=1)
        data.append(float(value))

if st.button('Predict'):
    # Convert to NumPy array and reshape for the model
    data = np.array(data).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(data)[0]
    result = 'Heart disease' if prediction == 1 else 'No heart disease'
    
    st.write(f"Prediction Result: {result}")