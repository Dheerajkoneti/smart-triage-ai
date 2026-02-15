def recommend_department(symptoms, risk):

    symptoms = symptoms.lower()

    if risk == "High":
        return "Emergency"

    if "chest pain" in symptoms:
        return "Cardiology"

    if "seizure" in symptoms:
        return "Neurology"

    return "General Medicine"
