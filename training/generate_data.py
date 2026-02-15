import pandas as pd
import random
import os

os.makedirs("../data", exist_ok=True)

data = []

for i in range(1000):
    age = random.randint(1, 90)
    gender = random.choice(["Male", "Female"])
    bp = random.randint(90, 200)
    hr = random.randint(60, 150)
    temp = round(random.uniform(97, 104), 1)

    symptoms_list = ["fever", "cough", "chest pain", "headache", "seizure"]
    conditions_list = ["diabetes", "hypertension", "none"]

    symptoms = random.choice(symptoms_list)
    condition = random.choice(conditions_list)

    # Risk Logic
    if bp > 180 or hr > 130 or temp > 103 or symptoms == "chest pain":
        risk = "High"
    elif bp > 140 or hr > 100 or temp > 101:
        risk = "Medium"
    else:
        risk = "Low"

    data.append([
        i, age, gender, symptoms, bp, hr, temp, condition, risk
    ])

columns = [
    "Patient_ID", "Age", "Gender", "Symptoms",
    "Blood_Pressure", "Heart_Rate", "Temperature",
    "Pre_Existing_Conditions", "Risk_Level"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("../data/synthetic_data.csv", index=False)

print("Synthetic dataset generated successfully!")
