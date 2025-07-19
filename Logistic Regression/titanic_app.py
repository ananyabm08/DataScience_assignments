import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('titanic_logreg_model.pkl', 'rb'))

st.title("Titanic Survival Prediction App")

st.markdown("""
Enter the passenger details below to predict their survival probability.
""")

# Input fields
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 80, 25)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', 0, 10, 0)
parch = st.number_input('Number of Parents/Children Aboard', 0, 10, 0)
fare = st.slider('Fare', 0.0, 500.0, 50.0)
embarked = st.selectbox('Port of Embarkation', ['Q', 'S'])
pclass = st.selectbox('Passenger Class', [1, 2, 3])

# Convert inputs to match model format
sex = 1 if sex == 'male' else 0
embarked_q = 1 if embarked == 'Q' else 0
embarked_s = 1 if embarked == 'S' else 0
pclass_2 = 1 if pclass == 2 else 0
pclass_3 = 1 if pclass == 3 else 0

# Final input array
features = np.array([[sex, age, sibsp, parch, fare, embarked_q, embarked_s, pclass_2, pclass_3]])

# Predict
if st.button('Predict'):
    prediction = model.predict(features)
    result = 'Survived' if prediction[0] == 1 else 'Did Not Survive'
    st.subheader(f"Prediction: {result}")
