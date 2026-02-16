import joblib
import numpy as np
import pandas as pd
import csv
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from utils.department_mapper import recommend_department
from utils.preprocessing import preprocess_input
from utils.explainability import get_feature_importance
from google.genai import Client
import json
import math
import sqlite3
from flask import session, redirect, url_for
from flask import jsonify
from authlib.integrations.flask_client import OAuth
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
DB_PATH = "/tmp/database.db"
load_dotenv()
client = Client(api_key=os.getenv("GOOGLE_API_KEY"))
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)
from functools import wraps

def roles_required(allowed_roles):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "role" not in session or session["role"] not in allowed_roles:
                return redirect("/login")
            return f(*args, **kwargs)
        return wrapper
    return decorator
import math
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT,
        role TEXT,
        is_active INTEGER DEFAULT 1
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        hospital_id TEXT,
        date TEXT,
        status TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        age INTEGER,
        gender TEXT,
        symptoms TEXT,
        bp REAL,
        hr REAL,
        temp REAL,
        risk_level TEXT,
        department TEXT,
        contact_status TEXT,
        assigned_doctor INTEGER,
        doctor_status TEXT,
        user_email TEXT
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        hospital TEXT,
        specialization TEXT,
        status TEXT DEFAULT 'Active'
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hospitals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        department TEXT,
        latitude REAL,
        longitude REAL,
        emergency_available TEXT,
        icu_beds INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        date DATE DEFAULT CURRENT_DATE
    )
    """)


    conn.commit()
    conn.close()
init_db()
# =========================================
# LOAD MODEL SAFELY
# =========================================
try:
    model = joblib.load("models/triage_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    print("✅ AI Model & Encoder Loaded Successfully")
except Exception as e:
    print("❌ Model Loading Failed:", e)
    model = None
    label_encoder = None


# =========================================
# SMART RECOMMENDATION ENGINE
# =========================================
def generate_recommendations(risk_level, department):
    """Generates clinical advice based on AI risk assessment."""
    if risk_level == "High":
        return {
            "color": "red",
            "priority": "⚠ Immediate medical attention required",
            "doctor": f"{department} Specialist / Emergency Physician",
            "diet": "Light fluids, ORS, avoid heavy meals",
            "medicine": "Paracetamol (if fever), Aspirin (if cardiac suspected), Visit hospital immediately"
        }
    elif risk_level == "Medium":
        return {
            "color": "orange",
            "priority": "🟡 Consult doctor within 24 hours",
            "doctor": f"{department} Specialist",
            "diet": "Balanced diet, hydration, low salt",
            "medicine": "Basic symptomatic medication"
        }
    else:
        return {
            "color": "green",
            "priority": "🟢 Low Risk – Home care sufficient",
            "doctor": "General Physician (optional)",
            "diet": "Normal healthy diet, hydration, rest",
            "medicine": "No strong medication required"
        }

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in KM
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (
        math.sin(dLat/2) * math.sin(dLat/2) +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dLon/2) * math.sin(dLon/2)
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c
def rank_hospitals(user_lat, user_lng, department_needed, risk_level):

    with open("data/hospitals_multicity_70.json") as f:
        hospitals = json.load(f)

    ranked = []

    for hospital in hospitals:

        try:
            lat = float(hospital["latitude"])
            lng = float(hospital["longitude"])

            distance = calculate_distance(user_lat, user_lng, lat, lng)

            # Department match
            specialist_score = 1 if department_needed in hospital["departments"] else 0

            icu_available = hospital.get("icu", 0)
            emergency_available = hospital.get("emergency", 0)

            # 🚨 Emergency override for High risk
            if risk_level == "High":
                if specialist_score == 0 or icu_available == 0:
                    continue

            # 🎯 Smart weighted score
            score = (
                (1 / (distance + 1)) * 40 +
                specialist_score * 30 +
                icu_available * 5 +
                emergency_available * 3
            )

            ranked.append({
                "id": hospital["id"],
                "name": hospital["name"],
                "location": hospital["location"],
                "distance_km": round(distance, 2),
                "icu": icu_available,
                "emergency": emergency_available,
                "score": round(score, 2)
            })
        except Exception as e:
            print("Hospital ranking error:", e)

    ranked.sort(key=lambda x: x["score"], reverse=True)

    return ranked[:5]
# =========================================
# CORE NAVIGATION ROUTES
# =========================================
# This route opens the file you created: patient_dashboard.html
@app.route("/")
def home():
    """Renders the futuristic landing page."""
    return render_template("landing.html")

@app.route("/triage")
def triage():
    """Renders the dark-themed patient intake form."""
    return render_template("index.html")
@app.route("/find_hospitals", methods=["POST"])
def find_hospitals():
    try:
        data = request.get_json()

        user_lat = float(data.get("lat"))
        user_lng = float(data.get("lng"))
        department = data.get("department")
        risk_level = data.get("risk_level")

        if not department or not risk_level:
            return jsonify({"error": "Missing department or risk level"}), 400

        hospitals = rank_hospitals(user_lat, user_lng, department, risk_level)

        return jsonify(hospitals)

    except Exception as e:
        print("Find hospitals error:", e)
        return jsonify({"error": "Internal server error"}), 500
@app.route("/auto_book_hospital", methods=["POST"])
def auto_book_hospital():

    if "role" not in session or session["role"] != "patient":
        return jsonify({"message": "Unauthorized"}), 403

    data = request.get_json()
    hospital_id = data.get("hospital_id")

    patient_id = session.get("user_id")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Insert appointment
    cursor.execute("""
        INSERT INTO appointments (patient_id, hospital_id, date, status)
        VALUES (?, ?, DATE('now'), ?)
    """, (patient_id, hospital_id, "Emergency"))

    conn.commit()
    conn.close()

    return jsonify({"message": "🚑 Emergency Appointment Booked!"})
# =========================================
# UPLOAD DOCUMENT PAGE
# =========================================
@app.route("/upload", methods=["GET", "POST"])
def upload_document():

    if request.method == "POST":
        file = request.files["document"]

        if file:
            file.save("uploads/" + file.filename)
            return "File uploaded successfully!"

    return render_template("upload.html")
@app.route("/authorize")
def authorize():
    token = google.authorize_access_token()

    resp = google.get("https://openidconnect.googleapis.com/v1/userinfo")
    user_info = resp.json()

    email = user_info["email"]
    name = user_info["name"]

    # 🔥 AUTO ROLE LOGIC
    if email.endswith("@hospital.com"):
        role = "doctor"
    else:
        role = "patient"

    # Save to DB if not exists
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    user_row = cursor.fetchone()

    if not user_row:
        cursor.execute("""
            INSERT INTO users (name, email, password, role)
            VALUES (?, ?, ?, ?)
        """, (name, email, "GOOGLE_AUTH", role))
        conn.commit()

        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user_row = cursor.fetchone()

    user_id = user_row[0]   # ✅ ALWAYS defined now

    conn.close()

    # Store in session
    session["user_id"] = user_id
    session["user"] = email
    session["name"] = name
    session["role"] = role
    if role == "doctor":
        return redirect("/doctor_dashboard")
    else:
        return redirect("/patient_dashboard")
@app.route("/login/google")
def google_login():
    return google.authorize_redirect(
        redirect_uri=url_for("authorize", _external=True)
    )
@app.route("/analytics")
@roles_required(["admin", "doctor"])
def analytics():

    import pandas as pd

    try:
        df = pd.read_csv("data/patient_data.csv")

        risk_counts = {
            "High": 0,
            "Medium": 0,
            "Low": 0
        }

        if "Risk_Level" in df.columns:
            counts = df["Risk_Level"].value_counts().to_dict()

            for key in risk_counts:
                risk_counts[key] = counts.get(key, 0)

    except Exception as e:
        print("Analytics Error:", e)
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}

    return render_template("analytics.html", risk_counts=risk_counts)

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")


    df = pd.read_csv("data/patient_data.csv")
    patients = df.to_dict(orient="records")
    return render_template("triage_dashboard.html", patients=patients)
@app.route("/notify/<patient_id>")
def notify_patient(patient_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE patients
        SET contact_status = 'Notified'
        WHERE patient_id = ?
    """, (patient_id,))

    conn.commit()
    conn.close()

    flash("Patient Notified Successfully!", "success")
    return redirect("/admin_dashboard")
@app.route("/escalate/<patient_id>")
def escalate_case(patient_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE patients
        SET doctor_status = 'Escalated'
        WHERE patient_id = ?
    """, (patient_id,))

    conn.commit()
    conn.close()

    flash("Case Escalated!", "warning")
    return redirect("/admin_dashboard")
# =========================================
# PREDICTION & DATA PERSISTENCE
# =========================================
@app.route("/predict", methods=["POST"])
def predict():
    """Processes intake data, runs AI, saves to database & auto assigns doctor."""
    if model is None or label_encoder is None:
        return "Critical Error: AI Model not loaded."

    try:
        # 1️⃣ Capture Form Data
        import uuid
        patient_id = "P" + str(uuid.uuid4())[:8]
        age = int(request.form["age"])
        gender = request.form["gender"]
        symptoms = request.form["symptoms"]
        bp = float(request.form["bp"])
        hr = float(request.form["hr"])
        temp = float(request.form["temp"])
        conditions = request.form["conditions"]

        # 2️⃣ AI Prediction
        input_data = preprocess_input(age, gender, symptoms, bp, hr, temp, conditions)
        prediction = model.predict(input_data)
        risk_level = label_encoder.inverse_transform(prediction)[0]
        confidence = float(np.max(model.predict_proba(input_data)) * 100)
        department = recommend_department(symptoms, risk_level)

        # 3️⃣ Auto Assign Doctor
        doctor_id = auto_assign_doctor(department)

        # 4️⃣ Save to DATABASE
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        user_email = session.get("user")

        cursor.execute("""
            INSERT INTO patients 
            (patient_id, age, gender, symptoms, bp, hr, temp, risk_level, department,
            contact_status, assigned_doctor, doctor_status, user_email)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            age,
            gender,
            symptoms,
            bp,
            hr,
            temp,
            risk_level,
            department,
            "Pending",
            doctor_id,
            "Active",   # 🔥 Make active immediately
            user_email
        ))

        # 🔥 Insert into predictions table for analytics
        cursor.execute("""
            INSERT INTO predictions (patient_id)
            VALUES (?)
        """, (patient_id,))

        conn.commit()
        conn.close()

        # 4️⃣.1 Sync to CSV Dataset (Realtime Update)
        csv_file = "data/patient_data.csv"
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Write header if file not exists
            if not file_exists:
                writer.writerow([
                    "Patient_ID", "Age", "Gender", "Symptoms",
                    "BP", "HR", "Temp", "Risk_Level", "Department"
                ])

            writer.writerow([
                patient_id,
                age,
                gender,
                symptoms,
                bp,
                hr,
                temp,
                risk_level,
                department
            ])
        # 5️⃣ Prepare UI Report
        raw_features = get_feature_importance(model, input_data)
        top_features = [
            {
                "name": feature.replace("_", " ").title(),
                "value": round(float(score) * 100, 1)
            }
            for feature, score in raw_features
        ]

        recommendations = generate_recommendations(risk_level, department)

        # Store session
        session["patient_data"] = {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "heart_rate": hr,
            "temperature": temp,
            "bp": bp,
            "risk_level": risk_level,
            "department": department,
            "confidence": round(confidence, 2)
        }

        return render_template(
            "result.html",
            patient_id=patient_id,
            risk_level=risk_level,
            confidence=round(confidence, 2),
            department=department,
            top_factors=top_features,
            recommendations=recommendations
        )

    except Exception as e:
        print("AI Prediction Error:", e)
        return f"Processing error: {str(e)}"
def auto_assign_doctor(department):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id FROM users 
        WHERE role='doctor'
    """)
    doctors = cursor.fetchall()

    if not doctors:
        conn.close()
        return None

    # Find doctor with least active patients
    cursor.execute("""
        SELECT assigned_doctor, COUNT(*) as total
        FROM patients
        WHERE doctor_status != 'Completed'
        GROUP BY assigned_doctor
        ORDER BY total ASC
        LIMIT 1
    """)
    result = cursor.fetchone()

    if result:
        doctor_id = result[0]
    else:
        doctor_id = doctors[0][0]

    conn.close()
    return doctor_id

# =========================================
# 🧠 MULTI-ROLE AI RESPONSE ENGINE
# =========================================
def generate_ai_response(message, role, patient_data=None):
    try:
        # =========================================
        # 🚨 EMERGENCY SAFETY LAYER
        # =========================================
        emergency_keywords = [
            "severe chest pain",
            "can't breathe",
            "difficulty breathing",
            "unconscious",
            "stroke",
            "seizure",
            "heart attack"
        ]

        if any(word in message.lower() for word in emergency_keywords):
            return "🚨 This may be life-threatening. Please call emergency services immediately."
        # =========================================
        # 🧠 ROLE BASED SYSTEM PROMPTS
        # =========================================
        if role == "triage":
            system_prompt = f"""
        You are MediTriage AI.

        Act as a triage assistant.
        Ask brief follow-up questions.
        Do not give final diagnosis.
        Be concise. Do not explain what you are doing.

        Patient Context:
        {patient_data}
        """

        elif role == "support":
            system_prompt = f"""
        You are MediTriage AI.

        Explain results briefly and clearly.
        Use simple language.
        Be concise.
        Do not over-explain.
        Patient Data:
        {patient_data}
        """
        elif role == "doctor":
            system_prompt = f"""
        You are MediTriage AI.
        Provide structured medical summary.
        Be professional and concise.
        Do not add unnecessary explanation.

        Patient Data:
        {patient_data}
        """
        else:
            system_prompt = "You are a concise medical AI assistant."
        # =========================================
        # 🧠 CHAT MEMORY
        # =========================================
        chat_history = session.get("chat_history", [])

        chat_history.append(f"User: {message}")

        # Combine history
        full_prompt = system_prompt + "\n\nConversation:\n" + "\n".join(chat_history[-6:])

        # =========================================
        # 🤖 GEMINI API CALL
        # =========================================
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt
        )
        reply = response.text
        # Save AI reply to history
        chat_history.append(f"AI: {reply}")
        session["chat_history"] = chat_history[-10:]
        return reply
    except Exception as e:
        print("🔥 GEMINI ERROR:", str(e))
        return "⚠ AI service unavailable. Please try again."
# =========================================
# 💬 CHAT API ROUTE
# =========================================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")
    role = data.get("role")
    language = data.get("language", "English")

    prompt = f"""
    You are a medical AI assistant in {role} mode.

    Respond naturally in {language}.
    Do NOT explain translation.
    Do NOT provide literal meanings.
    Only give the final response.

    User message:
    {message}
    """

    reply = generate_ai_response(prompt, role)

    return jsonify({"reply": reply})
@app.route("/patient_status", methods=["POST"])
def patient_status():
    try:
        data = request.get_json()
        pid = data.get("patient_id")

        if not pid:
            return jsonify({"error": "No Patient ID"}), 400

        pid = pid.strip().upper()

        import pandas as pd
        df = pd.read_csv("data/patient_data.csv")
        df.columns = df.columns.str.strip()
        df["Patient_ID"] = df["Patient_ID"].astype(str).str.upper()

        patient = df[df["Patient_ID"] == pid]

        if patient.empty:
            return jsonify({"error": "Patient not found"}), 404

        patient = patient.iloc[0]

        risk = patient["Risk_Level"]
        condition = patient["Pre_Existing_Conditions"]
        # 🔥 Map condition → department
        if condition in ["Hypertension", "Heart Disease"]:
            department = "Cardiology"
        elif condition == "None":
            department = "General"
        else:
            department = "General"

        import json
        with open("data/hospitals.json") as f:
            hospitals = json.load(f)

        hospitals = sorted(hospitals, key=lambda x: x["distance_km"])

        selected_hospital = None
        redirect = False
        distance = None

        for hospital in hospitals:
            if department in hospital["departments"]:
                beds_available = hospital["beds"].get(department, 0)

                if beds_available > 0:
                    selected_hospital = hospital
                    distance = hospital["distance_km"]
                    break
                else:
                    redirect = True

        if not selected_hospital:
            return jsonify({
                "patient_id": pid,
                "risk": risk,
                "department": department,
                "hospital": "None Available",
                "bed_status": "No Beds Available",
                "redirect": False
            })

        return jsonify({
            "patient_id": pid,
            "risk": risk,
            "department": department,
            "hospital": selected_hospital["name"],
            "distance": distance,
            "bed_status": "Beds Available",
            "redirect": redirect
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500
from flask import flash

from flask import flash

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):

            # ✅ Store session properly
            session["user_id"] = user[0]
            session["name"] = user[1]      # Full Name
            session["user"] = user[2]      # Email
            session["role"] = user[4]

            # 🔥 Role Based Redirect
            if user[4] == "patient":
                return redirect("/patient_dashboard")
            elif user[4] == "doctor":
                return redirect("/doctor_dashboard")
            elif user[4] == "admin":
                return redirect("/admin_dashboard")

        else:
            flash("Invalid Credentials", "error")

    return render_template("login.html")
@app.route("/patient_dashboard")
def patient_dashboard():

    if "role" not in session or session["role"] != "patient":
        return redirect("/login")

    user_email = session.get("user")
    user_id = session.get("user_id")   # ✅ VERY IMPORTANT

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 🔹 Get user name
    cursor.execute("SELECT name FROM users WHERE email = ?", (user_email,))
    user_data = cursor.fetchone()
    user_name = user_data[0] if user_data else "Unknown"

    # 🔹 Get triage history
    cursor.execute("""
        SELECT patient_id, risk_level, department, symptoms, bp, hr, temp
        FROM patients
        WHERE user_email = ?
        ORDER BY rowid DESC
    """, (user_email,))

    records = cursor.fetchall()

    triage_history = []
    for row in records:
        triage_history.append({
            "patient_id": row[0],
            "risk_level": row[1],
            "department": row[2],
            "symptoms": row[3],
            "bp": row[4],
            "hr": row[5],
            "temp": row[6]
        })

    latest_patient = triage_history[0] if triage_history else None

    # ✅ FIXED APPOINTMENT FILTER
    cursor.execute("""
        SELECT id, hospital_id, date, status
        FROM appointments
        WHERE patient_id = ?
        ORDER BY id DESC
    """, (user_id,))   # ✅ MUST BE user_id

    appointment_rows = cursor.fetchall()
    conn.close()

    # 🔹 Load hospitals
    import json
    with open("data/hospitals.json", "r") as f:
        hospitals = json.load(f)

    appointments = []

    for appt in appointment_rows:
        hospital_id = appt[1]

        hospital_info = next(
            (h for h in hospitals if int(h["id"]) == int(hospital_id)),
            None
        )

        if hospital_info:
            appointments.append({
                "appointment_id": appt[0],
                "hospital_name": hospital_info["name"],
                "hospital_location": hospital_info["location"],
                "distance_km": hospital_info.get("distance_km", "N/A"),
                "appointment_date": appt[2],
                "status": appt[3]
            })

    return render_template(
        "patient_dashboard.html",
        user_name=user_name,
        triage_history=triage_history,
        patient=latest_patient,
        hospitals=hospitals,
        appointments=appointments
    )
@app.route("/find_nearby_hospitals")
def find_nearby_hospitals():

    # 🔒 Only patient can access
    if "role" not in session or session["role"] != "patient":
        return redirect("/login")

    # 📍 Get user location
    user_lat = request.args.get("lat")
    user_lon = request.args.get("lon")

    if not user_lat or not user_lon:
        return "Location not provided"

    try:
        user_lat = float(user_lat)
        user_lon = float(user_lon)
    except ValueError:
        return "Invalid location format"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 🔥 Fetch hospital dataset
    cursor.execute("""
        SELECT id, name, location, latitude, longitude,
               emergency, icu, general
        FROM hospitals
    """)
    hospitals = cursor.fetchall()

    hospital_list = []

    for hospital in hospitals:
        try:
            hospital_id = hospital[0]
            name = hospital[1]
            location = hospital[2]
            lat = float(hospital[3])
            lon = float(hospital[4])
            emergency_beds = int(hospital[5])
            icu_beds = int(hospital[6])
            general_beds = int(hospital[7])

            # 📏 Distance Calculation
            distance = calculate_distance(user_lat, user_lon, lat, lon)

            # 🧠 AI Smart Scoring
            score = 0

            # 40% Distance weight
            score += max(0, 100 - distance) * 0.4  

            # 30% Emergency weight
            score += emergency_beds * 2  

            # 20% ICU weight
            score += icu_beds * 3  

            # 10% General beds
            score += general_beds * 1  

            hospital_list.append({
                "id": hospital_id,
                "name": name,
                "location": location,
                "distance_km": round(distance, 2),
                "emergency_beds": emergency_beds,
                "icu_beds": icu_beds,
                "general_beds": general_beds,
                "score": round(score, 2)
            })

        except Exception as e:
            print("Hospital processing error:", e)

    conn.close()

    # 🔥 Sort by AI Score
    hospital_list.sort(key=lambda x: x["score"], reverse=True)

    # 🎯 Return Top 5
    top_hospitals = hospital_list[:5]

    return render_template(
        "nearby_hospitals.html",
        hospitals=top_hospitals
    )
@app.route("/update_patient_status/<int:pid>", methods=["POST"])
def update_patient_status(pid):

    if "role" not in session or session["role"] != "doctor":
        return redirect("/login")

    new_status = request.form.get("status")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE patients
        SET doctor_status = ?
        WHERE id = ?
    """, (new_status, pid))

    conn.commit()
    conn.close()

    return redirect("/doctor_dashboard")
@app.route("/ai_suggestion/<patient_id>")
def ai_suggestion(patient_id):

    if session.get("role") != "doctor":
        return jsonify({"error": "Unauthorized"}), 403

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT symptoms, risk_level, department, bp, hr, temp
        FROM patients
        WHERE patient_id = ?
    """, (patient_id,))

    patient = cursor.fetchone()
    conn.close()

    if not patient:
        return jsonify({"error": "Not found"}), 404

    symptoms, risk, dept, bp, hr, temp = patient

    prompt = f"""
    Patient Symptoms: {symptoms}
    Risk Level: {risk}
    Department: {dept}
    BP: {bp}
    HR: {hr}
    Temp: {temp}

    Provide:
    1. Possible Differential Diagnosis
    2. Suggested Lab Tests
    3. Recommended Next Steps
    """

    reply = generate_ai_response(prompt, role="doctor")

    return jsonify({"suggestion": reply})
@app.route("/admin_dashboard")
def admin_dashboard():

    if "role" not in session or session["role"] != "admin":
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ============================
    # 📊 SYSTEM COUNTS
    # ============================

    cursor.execute("SELECT COUNT(*) FROM doctors")
    total_doctors = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM appointments")
    total_appointments = cursor.fetchone()[0]

    # Risk counts
    cursor.execute("SELECT COUNT(*) FROM patients WHERE risk_level='High'")
    high_risk_cases = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM patients WHERE risk_level='Medium'")
    medium_risk_cases = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM patients WHERE risk_level='Low'")
    low_risk_cases = cursor.fetchone()[0]

    # Departments count
    cursor.execute("SELECT COUNT(DISTINCT specialization) FROM doctors")
    departments_count = cursor.fetchone()[0]

    # Today's AI predictions
    cursor.execute("""
        SELECT COUNT(*) FROM predictions 
        WHERE DATE(date) = DATE('now')
    """)
    ai_predictions_today = cursor.fetchone()[0]

    # ============================
    # 👨‍⚕️ DOCTOR TABLE
    # ============================
    cursor.execute("""
        SELECT id, name, hospital, specialization, status
        FROM doctors
        ORDER BY id DESC
    """)
    doctors = cursor.fetchall()

    # ============================
    # 🧑‍🤝‍🧑 PATIENT TABLE
    # ============================
    cursor.execute("""
        SELECT patient_id, symptoms, risk_level, assigned_doctor
        FROM patients
        ORDER BY id DESC
    """)
    patients = cursor.fetchall()

    # ============================
    # 📈 RISK DISTRIBUTION (SAFE)
    # ============================
    risk_distribution = {
        "High": high_risk_cases,
        "Medium": medium_risk_cases,
        "Low": low_risk_cases
    }

    # ============================
    # 🚨 URGENT CASES
    # ============================
    cursor.execute("""
        SELECT patient_id, symptoms
        FROM patients
        WHERE risk_level='High'
        ORDER BY id DESC
        LIMIT 5
    """)
    urgent_cases = cursor.fetchall()

    conn.close()

    return render_template(
        "admin_dashboard.html",
        total_doctors=total_doctors,
        total_patients=total_patients,
        total_appointments=total_appointments,
        high_risk_cases=high_risk_cases,
        medium_risk_cases=medium_risk_cases,
        low_risk_cases=low_risk_cases,
        departments_count=departments_count,
        ai_predictions_today=ai_predictions_today,
        doctors=doctors,
        patients=patients,
        risk_distribution=risk_distribution,
        urgent_cases=urgent_cases
    )
@app.route("/view_report/<patient_id>")
def view_report(patient_id):

    if "role" not in session or session["role"] != "patient":
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT patient_id, risk_level, department, symptoms, bp, hr, temp
        FROM patients
        WHERE patient_id = ?
    """, (patient_id,))

    record = cursor.fetchone()
    conn.close()

    if not record:
        return "Report not found"

    report = {
        "patient_id": record[0],
        "risk_level": record[1],
        "department": record[2],
        "symptoms": record[3],
        "bp": record[4],
        "hr": record[5],
        "temp": record[6]
    }

    return render_template("report_view.html", report=report)
@app.route("/add_doctor", methods=["GET", "POST"])
def add_doctor():
    if request.method == "POST":
        name = request.form["name"]
        hospital = request.form["hospital"]
        specialization = request.form["specialization"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO doctors (name, hospital, specialization)
            VALUES (?, ?, ?)
        """, (name, hospital, specialization))

        conn.commit()
        conn.close()

        return redirect("/admin_dashboard")

    return render_template("add_doctor.html")
@app.route("/doctor_dashboard")
def doctor_dashboard():

    if "role" not in session or session["role"] != "doctor":
        return redirect("/login")

    doctor_id = session.get("user_id")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Assigned Patients
    cursor.execute("""
        SELECT * FROM patients
        WHERE assigned_doctor = ?
    """, (doctor_id,))
    patients = cursor.fetchall()

    assigned_patients = len(patients)

    # Active Patients
    active_patients = len([p for p in patients if p["doctor_status"] == "Active"])

    # High Risk Patients
    high_risk_cases = len([p for p in patients if p["risk_level"] == "High"])

    # Urgent Patients (High Risk + Active)
    urgent_patients = [
        p for p in patients
        if p["risk_level"] == "High" and p["doctor_status"] == "Active"
    ]

    urgent_count = len(urgent_patients)

    conn.close()

    return render_template(
        "doctor_dashboard.html",
        patients=patients,
        assigned_patients=assigned_patients,
        active_patients=active_patients,
        high_risk_cases=high_risk_cases,
        urgent_patients=urgent_patients,
        urgent_count=urgent_count,
        doctor_name="Naveen",   # Replace with DB later
        doctor_hospital="Rainbow Hospital",
        doctor_specialization="General"
   )
# ==============================
# 🔐 MANAGE ROLES
# ==============================
@app.route("/admin/manage_roles")
def manage_roles():
    if "role" not in session or session["role"] != "admin":
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, role FROM users")
    users = cursor.fetchall()

    conn.close()

    return render_template("manage_roles.html", users=users)


# ==============================
# 📋 VIEW LOGS
# ==============================
@app.route("/admin/view_logs")
def view_logs():
    if "role" not in session or session["role"] != "admin":
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM logs ORDER BY created_at DESC LIMIT 50")
    logs = cursor.fetchall()

    conn.close()

    return render_template("view_logs.html", logs=logs)


# ==============================
# 🛑 DISABLE USER PAGE
# ==============================
@app.route("/admin/disable_user")
def disable_user_page():
    if "role" not in session or session["role"] != "admin":
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, role, is_active FROM users")
    users = cursor.fetchall()

    conn.close()

    return render_template("disable_user.html", users=users)
@app.route("/retrain_model", methods=["GET", "POST"])
def retrain_model():

    if request.method == "POST":
        # Your real ML training code here
        flash("Model retrained successfully!", "success")
        return redirect("/admin_dashboard")

    return render_template("retrain_model.html")
@app.route("/edit_doctor/<int:doctor_id>", methods=["GET", "POST"])
def edit_doctor(doctor_id):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if request.method == "POST":
        name = request.form["name"]
        hospital = request.form["hospital"]
        specialization = request.form["specialization"]

        cursor.execute("""
            UPDATE doctors
            SET name = ?, hospital = ?, specialization = ?
            WHERE id = ?
        """, (name, hospital, specialization, doctor_id))

        conn.commit()
        conn.close()

        flash("Doctor updated successfully!", "success")
        return redirect("/admin_dashboard")

    # GET request (load doctor data)
    cursor.execute("SELECT * FROM doctors WHERE id = ?", (doctor_id,))
    doctor = cursor.fetchone()
    conn.close()

    if not doctor:
        return "Doctor not found"

    return render_template("edit_doctor.html", doctor=doctor)
@app.route("/delete_doctor/<int:doctor_id>")
def delete_doctor(doctor_id):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM doctors WHERE id = ?", (doctor_id,))
    conn.commit()
    conn.close()

    flash("Doctor removed successfully!", "success")

    return redirect("/admin_dashboard")
@app.route("/reassign/<patient_id>", methods=["GET", "POST"])
def reassign_patient(patient_id):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if request.method == "POST":
        new_doctor = request.form["doctor_id"]

        cursor.execute("""
            UPDATE patients
            SET assigned_doctor = ?
            WHERE patient_id = ?
        """, (new_doctor, patient_id))

        conn.commit()
        conn.close()

        return redirect("/admin_dashboard")

    cursor.execute("SELECT * FROM doctors")
    doctors = cursor.fetchall()

    conn.close()

    return render_template(
        "reassign.html",
        patient_id=patient_id,
        doctors=doctors
    )
@app.route('/admin/edit_privileges/<int:user_id>', methods=['GET', 'POST'])
def edit_privileges(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if request.method == 'POST':
        new_role = request.form['role']

        cursor.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
        conn.commit()
        conn.close()

        return redirect('/admin_dashboard')

    # GET request
    cursor.execute("SELECT id, name, role FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
@app.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():

    if request.method == "POST":
        file = request.files.get("dataset")

        if file:
            file.save("data/" + file.filename)
            flash("Dataset uploaded successfully!", "success")
            return redirect("/admin_dashboard")

    return render_template("upload_dataset.html")
# ==============================
# 🔴 ACTUALLY DISABLE USER
# ==============================
@app.route("/admin/disable_user/<int:user_id>")
def disable_user(user_id):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

    return redirect("/admin/disable_user")
@app.route("/fix_users_table")
def fix_users_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1")
        conn.commit()
    except:
        pass

    conn.close()
    return "Users table updated!"
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")  # ✅ FIXED
        email = request.form.get("email")
        password = request.form.get("password")
        role = request.form.get("role")

        if not name or not email or not password or not role:
            flash("All fields are required", "error")
            return redirect("/register")

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO users (name, email, password, role)
                VALUES (?, ?, ?, ?)
            """, (name, email, hashed_password, role))

            conn.commit()
            flash("Account created successfully 🎉 Please login.", "success")

        except sqlite3.IntegrityError:
            flash("User already exists ❌", "error")
            return redirect("/register")

        conn.close()
        return redirect("/login")

    return render_template("register.html")
@app.route("/book_appointment", methods=["POST"])
def book_appointment():

    if "role" not in session or session["role"] != "patient":
        return redirect("/login")

    # 🔥 Convert to INTEGER
    hospital_id = int(request.form["hospital_id"])
    appointment_date = request.form["date"]
    patient_id = session.get("user_id")

    if not patient_id:
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO appointments (patient_id, hospital_id, date, status)
        VALUES (?, ?, ?, ?)
    """, (patient_id, hospital_id, appointment_date, "Pending"))

    conn.commit()
    conn.close()

    flash("Appointment Booked Successfully!", "success")
    return redirect("/patient_dashboard")
# =========================================
# APP EXECUTION
# =========================================
if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)