import pandas as pd

def preprocess_input(age, gender, symptoms, bp, hr, temp, conditions):

    gender_encoded = 1 if gender.lower() == "male" else 0

    data = {
        "Age": [age],
        "Gender": [gender_encoded],
        "Blood_Pressure": [bp],
        "Heart_Rate": [hr],
        "Temperature": [temp]
    }

    return pd.DataFrame(data)
