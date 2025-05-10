import os
import uuid
import datetime
from functools import wraps

from flask import Flask, request, jsonify, g
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
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://rakeshbanda:root@localhost/sepsis_db")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALGORITHM = "HS256"
JWT_EXP_MINUTES = int(os.getenv("JWT_EXP_MINUTES", "60"))

engine = create_engine(DATABASE_URL, echo=False, future=True)
Base = declarative_base()
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=JWT_EXP_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")


# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------

class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    def __repr__(self):
        return f"<Role {self.name}>"


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)  # hashed
    full_name = Column(String)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    role = relationship("Role")

    def __repr__(self):
        return f"<User {self.username} ({self.role.name})>"


class Patient(Base):
    __tablename__ = "patients"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    dob = Column(Date, nullable=False)
    gender = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    notes = relationship("ClinicalNote", back_populates="patient", cascade="all, delete-orphan")
    predictions = relationship("PredictionCache", back_populates="patient", cascade="all, delete-orphan")
    histories = relationship("MedicalHistory", back_populates="patient", cascade="all, delete-orphan")


class ClinicalNote(Base):
    __tablename__ = "clinical_notes"
    id = Column(Integer, primary_key=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    author = Column(String, nullable=False)
    role = Column(String)
    note = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    patient = relationship("Patient", back_populates="notes")


class PredictionCache(Base):
    __tablename__ = "prediction_cache"
    id = Column(Integer, primary_key=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    sepsis_risk = Column(Float, nullable=False)
    alert = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    requested_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))

    patient = relationship("Patient", back_populates="predictions")


class MedicalHistory(Base):
    __tablename__ = "medical_history"
    id = Column(Integer, primary_key=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    visit_date = Column(Date, nullable=False)
    diagnosis = Column(Text)
    medications = Column(Text)
    notes = Column(Text)
    document_path = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    patient = relationship("Patient", back_populates="histories")


# ---------------------------------------------------------------------------
# Flask application factory & middleware
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # ------------------------------------------------------
    # Request lifecycle helpers
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Auth decorators
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Routes
    # ------------------------------------------------------

    @app.route("/register", methods=["POST"])
    def register():
        data = request.json
        username = data.get("username")
        password = data.get("password")
        role_name = data.get("role", "patient_watcher")
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
        token = create_jwt_token(str(user.id), user.role.name)
        return jsonify({"token": token, "role": user.role.name})

    # Example protected route
    @app.route("/patients/<uuid:patient_id>")
    @jwt_required(allowed_roles=["doctor", "nurse", "patient_watcher"])
    def get_patient(patient_id):
        patient = g.db.query(Patient).get(patient_id)
        if not patient:
            return jsonify({"error": "Patient not found"}), 404
        result = {
            "id": str(patient.id),
            "name": patient.name,
            "age": patient.age,
            "dob": patient.dob.isoformat(),
            "gender": patient.gender,
        }
        return jsonify(result)

    return app


# ---------------------------------------------------------------------------
# Database initialization helper
# ---------------------------------------------------------------------------

def init_db():
    Base.metadata.create_all(engine)
    session = SessionLocal()
    # Seed default roles if they do not exist
    for role_name in ["doctor", "nurse", "patient_watcher"]:
        if not session.query(Role).filter_by(name=role_name).first():
            session.add(Role(name=role_name))
    session.commit()
    session.close()


if __name__ == "__main__":
    init_db()
    app = create_app()
    app.run(host="0.0.0.0", port=5002, debug=True)
