from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import os
from twilio.rest import Client
from dotenv import load_dotenv
import re
import pandas as pd
import requests
from pathlib import Path
import os
from flask import g
from flask import g, Flask, request, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import bcrypt
import jwt
import datetime
from functools import wraps
from flask_cors import CORS
from sqlalchemy import (
    create_engine, Column, String, Integer, Date, DateTime, Text, Boolean, Float,
    ForeignKey, Enum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, scoped_session
from sqlalchemy.exc import IntegrityError
import bcrypt
import jwt
from dotenv import load_dotenv
from pathlib import Path
# At the top of app_new.py
from auth_jwt import (
    Base, Role, User, 
    gen_uuid, hash_password, verify_password, 
    create_jwt_token, SessionLocal
)
JWT_SECRET = os.getenv("JWT_SECRET", "your-strong-secret-key-here")  # Always use env var in production
JWT_ALGORITHM = "HS256"
# from auth_jwt import jwt_required, SessionLocal
# from auth_jwt import ClinicalNote
load_dotenv(dotenv_path=Path("/Users/rakeshbanda/Documents/capstone project/.env"))
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(BASE_DIR, "sepsis_lstm.h5"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
patient_df = pd.read_csv(os.path.join(BASE_DIR, "synthetic_icu_data.csv"))

# === RAG PDF Chatbot Setup ===
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore

#added new
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def ask_llm(prompt):
    return chat_model.invoke([HumanMessage(content=prompt)]).content


def build_vitals_prompt(patient_id, vitals, original_query):
    return f"""
You are a helpful medical assistant. A user is asking about vitals for Patient ID {patient_id}. 
Here are the vitals retrieved from the database:

- Heart Rate: {vitals['heart_rate']} bpm
- Blood Pressure: {vitals['bp']} mmHg
- Oxygen Saturation: {vitals['o2_sat']}%
- Respiratory Rate: {vitals['resp_rate']} bpm
- Temperature: {vitals['temperature']} °F

Please answer the user's query in a friendly, professional tone using this information.

User Query: {original_query}
"""

##end of newly added

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))


embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
retriever = vectorstore.as_retriever()
chat_model = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=retriever)

Base = declarative_base()

class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)  # hashed
    full_name = Column(String)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    role = relationship("Role")

def gen_uuid() -> str:
    return str(uuid.uuid4())

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_jwt_token(user_id: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=60),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def decode_jwt_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")
def jwt_required(allowed_roles=None):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                auth_header = request.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    return jsonify({"error": "Missing or invalid auth header"}), 401
                token = auth_header.split(" ", 1)[1]
                try:
                    payload = decode_jwt_token(token)
                except ValueError as e:
                    return jsonify({"error": str(e)}), 401

                user_id = payload["sub"]
                role_name = payload["role"]

                if allowed_roles and role_name not in allowed_roles:
                    return jsonify({"error": "Forbidden"}), 403

                g.current_user = g.db.query(User).get(user_id)
                return fn(*args, **kwargs)
            return wrapper
        return decorator
# # Explicitly load .env with absolute path
# env_path = Path("/Users/rakeshbanda/Documents/capstone project/.env")
# if not env_path.exists():
#     raise FileNotFoundError(f".env file not found at {env_path}")

# load_dotenv(dotenv_path=env_path, override=True)

# # Debug print - will show in your Flask console
# print("\n=== Loaded Environment Variables ===")
# print(f"API_KEY: {'****' + os.getenv('OPENROUTER_API_KEY')[-4:] if os.getenv('OPENROUTER_API_KEY') else 'NOT FOUND'}")
# print(f"MODEL: {os.getenv('OPENROUTER_MODEL')}")
# print(f"REFERER: {os.getenv('OPENROUTER_REFERER')}\n")


# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

#  # Replace with env variable for security
# OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
# OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER")

# === INIT FLASK APP ===
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# # === Load Sepsis Model & Scaler ===
# model = tf.keras.models.load_model("/Users/rakeshbanda/Documents/capstone project/sepsis_lstm.h5")
# scaler = joblib.load("/Users/rakeshbanda/Documents/capstone project/scaler.joblib")

# # === Load Patient CSV for chatbot ===
# patient_df = pd.read_csv("synthetic_icu_data.csv")
features_to_scale = ['heart_rate', 'sbp', 'lactate', 'resp_rate']

# === Twilio Setup ===
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")
RECIPIENT_PHONE = os.getenv("RECIPIENT_PHONE")

# === Helper Functions ===

def send_sms_alert(patient_id, sepsis_risk):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    message = f"🚨 Sepsis Alert! Patient {patient_id} has a risk score of {sepsis_risk * 100:.1f}%. Immediate action required."
    try:
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE,
            to=RECIPIENT_PHONE
        )
        print(f"✅ SMS alert sent to {RECIPIENT_PHONE}")
    except Exception as e:
        print(f"❌ Error sending SMS: {e}")

def get_patient_vitals(patient_id):
    # if str(patient_id) in patient_df['patient_id'].astype(str).values:
    #     row = patient_df[patient_df['patient_id'] == int(patient_id)].iloc[0]
    #     vitals = {f: float(row[f]) for f in features_to_scale}
    #     return vitals
    # return None
    icu_data = pd.read_csv("synthetic_icu_data.csv")
    patient_rows = icu_data[icu_data['patient_id'] == patient_id]
    
    if not patient_rows.empty:
        # Get the row with the highest time (i.e., latest vitals)
        latest_row = patient_rows.sort_values("hour", ascending=False).iloc[0]
        return latest_row.to_dict()
    
    return None

def chatbot_response(user_query):
    # === If asking about vitals ===
    if "vitals" in user_query.lower():
        match = re.search(r"\b\d+\b", user_query)
        if match:
            patient_id = int(match.group(0))
            vitals = get_patient_vitals(patient_id)
            if vitals:
                prompt = build_vitals_prompt(patient_id, vitals, user_query)
                try:
                    response = ask_llm(prompt)
                    return response
                except Exception as e:
                    return f"LLM error while generating vitals response: {str(e)}"
            else:
                return f"No vitals found for Patient ID {patient_id}."
        else:
            return "Please provide a valid patient ID."

    # === Otherwise use LLM ===
    try:
        result = qa_chain.invoke(user_query)
        return result["result"]
    except Exception as e:
        print(f"RAG chatbot error: {str(e)}")
        return "Sorry, I couldn't process that question with the knowledge base."

# === Routes ===

@app.route('/')
def home():
    return "ICU Clinical Decision Support + Chatbot API is running"

@app.route('/predict', methods=['POST'])
# @jwt_required(allowed_roles=["doctor", "nurse"])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Use get_json() instead of .json
        
        # Validate incoming data
        if not data or 'vitals' not in data:
            return jsonify({'error': 'Missing vitals data'}), 400
            
        vitals = np.array(data['vitals'])
        
        # Validate shape - expecting (timesteps, 4)
        if len(vitals.shape) != 2 or vitals.shape[1] != 4:
            return jsonify({'error': 'Invalid data shape. Expected (n_timesteps, 4)'}), 400
        
        # Process data
        vitals_scaled = scaler.transform(vitals.reshape(-1, 4)).reshape(1, -1, 4)
        prediction = model.predict(vitals_scaled)[0][0]
        
        # Handle alert
        alert = prediction > 0.5
        patient_id = data.get('patient_id', 'unknown')
        
        if alert:
            send_sms_alert(patient_id, prediction)
        
        return jsonify({
            'sepsis_risk': float(prediction),
            'alert': bool(alert),  # Ensure boolean serialization
            'patient_id': patient_id,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    user_input = data.get("query", "")
    response = chatbot_response(user_input)
    return jsonify({"response": response})
@app.route("/test_openrouter")
def test_openrouter():
    try:
        # Get fresh values from environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("OPENROUTER_MODEL")
        referer = os.getenv("OPENROUTER_REFERER")
        
        if not all([api_key, model, referer]):
            return jsonify({"error": "Missing environment variables"}), 500

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": referer,
            "X-Model": model,
            "User-Agent": "ICU-CDSS-Test/1.0"
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}]
            },
            timeout=10
        )

        return jsonify({
            "status": response.status_code,
            "headers_sent": {k: "****" if k == "Authorization" else v 
                           for k, v in headers.items()},
            "response": response.json() if response.ok else response.text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/verify_key")
def verify_key():
    """Directly tests the API key with minimal requirements"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return jsonify({"error": "No API key found in environment"}), 401
        
    response = requests.post(
        "https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=5
    )
    return jsonify({
        "key_valid": response.ok,
        "status": response.status_code,
        "response": response.json() if response.ok else response.text
    })

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    role_name = data.get("role", "patient_watcher")  # Default to "patient_watcher"
    full_name = data.get("full_name", "")

    if not all([username, password]):
        return jsonify({"error": "Username and password required"}), 400

    role = g.db.query(Role).filter_by(name=role_name).first()
    if not role:
        return jsonify({"error": "Invalid role"}), 400

    user = User(id=gen_uuid(), username=username, password=hash_password(password), role=role, full_name=full_name)
    g.db.add(user)
    try:
        g.db.commit()
    except IntegrityError:
        g.db.rollback()
        return jsonify({"error": "Username already exists"}), 409

    token = create_jwt_token(str(user.id), role.name)
    return jsonify({"token": token, "role": role.name})

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    user = g.db.query(User).filter_by(username=username).first()

    if not user or not verify_password(password, user.password):
        return jsonify({"error": "Invalid credentials"}), 401

    # If login is successful, generate the JWT token
    token = create_jwt_token(str(user.id), user.role.name)
    return jsonify({"token": token, "role": user.role.name})

# Add database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://rakeshbanda:root@localhost/sepsis_db")
engine = create_engine(DATABASE_URL)
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))

# Add before_request handler
@app.before_request
def open_session():
    g.db = SessionLocal()

@app.teardown_request
def close_session(exception=None):
    db = g.pop("db", None)
    if db is not None:
        if exception:
            db.rollback()
        else:
            db.commit()
        db.close()
# @app.route('/get_notes/<uuid:patient_id>', methods=['GET'])
# # @jwt_required(allowed_roles=["doctor", "nurse", "patient_watcher"])
# def get_notes(patient_id):
#     session = SessionLocal()
#     try:
#         notes = session.query(ClinicalNote)\
#                        .filter_by(patient_id=patient_id)\
#                        .order_by(ClinicalNote.timestamp.desc())\
#                        .all()
#         return jsonify([{
#             "id": note.id,
#             "author": note.author,
#             "role": note.role,
#             "note": note.note,
#             "timestamp": note.timestamp.isoformat()
#         } for note in notes])
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         session.close()

    
# === Run the App ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
