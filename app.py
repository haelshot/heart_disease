import streamlit as st
import joblib
import numpy as np

try:
    model = joblib.load('model.pkl')
except:
    st.error("Failed to load the model. Make sure 'model.pkl' is in the same directory as this script.")
    st.stop()

variables = {
    "age": {"type": "number", "default": 50},
    "sex": {"type": "select", "options": ["female", "male"], "default": "male"},
    "cp": {"type": "select", "options": ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"], "default": "typical angina"},
    "trestbps": {"type": "number", "default": 120},
    "chol": {"type": "number", "default": 200},
    "fbs": {"type": "select", "options": ["lower than 120mg/ml", "greater than 120mg/ml"], "default": "lower than 120mg/ml"},
    "restecg": {"type": "select", "options": ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"], "default": "normal"},
    "thalach": {"type": "number", "default": 150},
    "exang": {"type": "select", "options": ["no", "yes"], "default": "no"},
    "oldpeak": {"type": "number", "default": 1},
    "slope": {"type": "select", "options": ["upsloping", "flat", "downsloping"], "default": "upsloping"},
    "ca": {"type": "number", "default": 0},
    "thal": {"type": "select", "options": ["normal", "fixed defect", "reversable defect"], "default": "normal"}
}

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
st.write("The prediction can either be `heart disease` or `No heart disease`")

if 'changed' not in st.session_state:
    st.session_state.changed = False

def set_changed():
    st.session_state.changed = True

data = []
for var, info in variables.items():
    if info["type"] == "select":
        value = st.selectbox(
            var.capitalize(), 
            info["options"], 
            index=info["options"].index(info["default"]),
            key=var,
            on_change=set_changed
        )
        data.append(mappings[var][value])
    else:
        value = st.number_input(
            var.capitalize(), 
            value=info["default"], 
            step=1,
            key=var,
            on_change=set_changed
        )
        data.append(float(value))

if st.button('Predict', disabled=not st.session_state.changed):
    data = np.array(data).reshape(1, -1)
    
    try:
        prediction = model.predict(data)[0]
        result = 'Heart disease' if prediction == 1 else 'No heart disease'
        st.write(f"Prediction Result: {result}")
    except:
        st.error("An error occurred during prediction. Please try again.")

if not st.session_state.changed:
    st.info("Input your values for prediction button to become active")