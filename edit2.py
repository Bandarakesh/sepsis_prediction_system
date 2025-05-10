import pandas as pd
import streamlit as st
import plotly.express as px
import requests
import random
from faker import Faker
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import sql
import uuid

# Initialize tools
fake = Faker()

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="sepsis_db",
        user="rakeshbanda",
        password="root"
    )

# ==================== DATA PREPARATION ====================
@st.cache_data
def load_vitals_data():
    """Load and cache the CSV vitals data"""
    return pd.read_csv('/Users/rakeshbanda/Documents/capstone project/synthetic_icu_data.csv')

def generate_patient_profile(patient_id):
    """Create realistic patient demographics for each ID"""
    gender = random.choice(['Male', 'Female'])
    admission_date = datetime.now() - timedelta(days=random.randint(1, 30))
    
    return {
        'patient_id': patient_id,
        'name': fake.name_male() if gender == 'Male' else fake.name_female(),
        'age': random.randint(18, 90),
        'gender': gender,
        'admission_date': admission_date.strftime('%Y-%m-%d'),
        'condition': random.choice(['Pneumonia', 'Sepsis', 'COVID-19', 'Heart Failure', 'Post-Op Recovery']),
        'allergies': random.choice(['None', 'Penicillin', 'NSAIDs', 'Sulfa', 'Latex'])
    }

# ==================== NOTES MANAGEMENT ====================
def get_patient_notes(patient_id):
    """Retrieve notes from database"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT doctor_notes, nurse_notes, last_updated 
            FROM patient_notes 
            WHERE patient_id = %s
            """, (int(patient_id),))
        result = cur.fetchone()
        return {
            'doctor_notes': result[0] if result else "",
            'nurse_notes': result[1] if result else "",
            'last_updated': result[2] if result else None
        } if result else None
    finally:
        conn.close()

def save_patient_notes(patient_id, doctor_notes, nurse_notes):
    """Save notes to database"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO patient_notes (patient_id, doctor_notes, nurse_notes, last_updated)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (patient_id)
            DO UPDATE SET 
                doctor_notes = EXCLUDED.doctor_notes,
                nurse_notes = EXCLUDED.nurse_notes,
                last_updated = NOW()
            """, (int(patient_id), doctor_notes, nurse_notes))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

# ==================== PATIENT MANAGEMENT ====================
def add_new_patient_to_db(name, age, dob, gender):
    """Insert a new patient into the database with UUID"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        patient_id = str(uuid.uuid4())  # Generate new UUID
        cur.execute("""
            INSERT INTO patients (id, name, age, dob, gender, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            RETURNING id
            """, (patient_id, name, age, dob, gender))
        conn.commit()
        return patient_id
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None
    finally:
        conn.close()

def get_all_patients_from_db():
    """Retrieve all patients from database"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, name, age, dob, gender, created_at 
            FROM patients 
            ORDER BY created_at DESC
            """)
        return cur.fetchall()
    finally:
        conn.close()

def show_new_patient_form():
    """Form to add new patients matching the database schema"""
    with st.expander("➕ Add New Patient", expanded=False):
        with st.form("new_patient_form", clear_on_submit=True):  # Add clear_on_submit=True
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name*", help="Required field")
                age = st.number_input("Age*", min_value=0, max_value=120, step=1)
                
            with col2:
                dob = st.date_input("Date of Birth*", 
                                   max_value=datetime.now().date(),
                                   help="Required field")
                gender = st.selectbox("Gender", ["", "Male", "Female", "Other", "Unknown"])
            
            submitted = st.form_submit_button("Add Patient")
            if submitted:
                if not name or not age or not dob:
                    st.error("Please fill all required fields (*)")
                else:
                    patient_id = add_new_patient_to_db(
                        name=name,
                        age=age,
                        dob=dob,
                        gender=gender if gender else None
                    )
                    if patient_id:
                        st.success(f"Patient {name} added successfully!")
                        # Update session state
                        st.session_state.patient_profiles[patient_id] = {
                            'patient_id': patient_id,
                            'name': name,
                            'age': age,
                            'dob': dob.strftime('%Y-%m-%d'),
                            'gender': gender,
                            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M')
                        }
                        st.rerun()

# ==================== DASHBOARD LAYOUT ====================
def main():
    st.title("ICU Patient Monitoring Dashboard")
    
    # Load data
    df = load_vitals_data()
    
    # Generate patient profiles (once per session)
    if 'patient_profiles' not in st.session_state:
        st.session_state.patient_profiles = {
            pid: generate_patient_profile(pid) 
            for pid in df['patient_id'].unique()
        }
    
    # Initialize notes in session state
    if 'clinical_notes' not in st.session_state:
        st.session_state.clinical_notes = {}
    
    # Show new patient form
    show_new_patient_form()
    
    # Patient selection dropdown
    patient_options = [
        f"{pid} - {st.session_state.patient_profiles[pid]['name']}" 
        for pid in df['patient_id'].unique()
    ]
    selected_patient = st.selectbox("Select Patient", options=patient_options)
    patient_id = int(selected_patient.split(" - ")[0])
    patient_info = st.session_state.patient_profiles[patient_id]
    
    # Load notes if not already in session state
    if patient_id not in st.session_state.clinical_notes:
        db_notes = get_patient_notes(patient_id)
        st.session_state.clinical_notes[patient_id] = db_notes or {
            'doctor_notes': "",
            'nurse_notes': "",
            'last_updated': None
        }
    
    # ===== PATIENT INFO SECTION =====
    with st.expander("Patient Summary", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(patient_info['name'])
            st.write(f"**Age:** {patient_info['age']}")
            st.write(f"**Gender:** {patient_info['gender']}")
            st.write(f"**Admitted:** {patient_info['admission_date']}")
            
        with col2:
            st.write(f"**Condition:** {patient_info['condition']}")
            st.write(f"**Allergies:** {patient_info['allergies']}")
            st.write(f"**Patient ID:** {patient_id}")
    
    # ===== CLINICAL NOTES SECTION =====
    st.subheader("Clinical Documentation")
    
    tab1, tab2 = st.tabs(["Doctor's Notes", "Nurse's Notes"])
    
    with tab1:
        doctor_notes = st.text_area(
            "Physician Documentation",
            value=st.session_state.clinical_notes[patient_id]['doctor_notes'],
            height=200,
            key=f"doctor_notes_{patient_id}"
        )
    
    with tab2:
        nurse_notes = st.text_area(
            "Nursing Documentation",
            value=st.session_state.clinical_notes[patient_id]['nurse_notes'],
            height=200,
            key=f"nurse_notes_{patient_id}"
        )
    
    if st.button("Save All Notes"):
        if save_patient_notes(patient_id, doctor_notes, nurse_notes):
            st.session_state.clinical_notes[patient_id] = {
                'doctor_notes': doctor_notes,
                'nurse_notes': nurse_notes,
                'last_updated': datetime.now()
            }
            st.success("Notes saved successfully!")
        else:
            st.error("Failed to save notes")
    
    # ===== VITAL SIGNS SECTION =====
    st.subheader("Vital Signs Monitoring")
    
    # Filter vitals for selected patient
    patient_vitals = df[df['patient_id'] == patient_id]
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Current Vitals", "Sepsis Risk"])
    
    with tab1:
        selected_vitals = st.multiselect(
            "Select metrics to display",
            options=['heart_rate', 'sbp', 'resp_rate', 'lactate'],
            default=['heart_rate', 'sbp']
        )
        
        if selected_vitals:
            fig = px.line(
                patient_vitals, 
                x='hour', 
                y=selected_vitals,
                title=f"24-Hour Trend for {patient_info['name']}",
                labels={'hour': 'Hours Since Admission', 'value': 'Value'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        latest = patient_vitals.iloc[-1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Heart Rate", f"{latest['heart_rate']} bpm")
            st.metric("Systolic BP", f"{latest['sbp']} mmHg")
            
        with col2:
            st.metric("Respiratory Rate", f"{latest['resp_rate']} /min")
            st.metric("Lactate", f"{latest['lactate']} mmol/L")
    
    with tab3:
        if st.button("Assess Sepsis Risk"):
            with st.spinner("Analyzing vitals..."):
                try:
                    vitals_data = patient_vitals[['heart_rate', 'sbp', 'lactate', 'resp_rate']].values
                    
                    if len(vitals_data) < 24:
                        st.warning(f"Only {len(vitals_data)} hours of data available. Using what we have.")
                    
                    response = requests.post(
                        'http://localhost:5001/predict',
                        json={
                            'vitals': vitals_data.tolist(),
                            'patient_id': patient_id
                        },
                        headers={'Content-Type': 'application/json'},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('status') == 'success':
                            risk = result['sepsis_risk']
                            
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric("Risk Score", f"{risk*100:.1f}%")
                            with col2:
                                st.progress(risk)
                            
                            if risk > 0.7:
                                st.error("""
                                🚨 Critical Sepsis Risk
                                - Immediate intervention required
                                - Consider broad-spectrum antibiotics
                                - Notify attending physician immediately
                                """)
                            elif risk > 0.5:
                                st.warning("""
                                ⚠️ Moderate Risk
                                - Close monitoring needed
                                - Consider blood cultures
                                - Reassess in 1 hour
                                """)
                            else:
                                st.success("""
                                ✅ Low Risk
                                - Continue routine monitoring
                                - Reassess if condition changes
                                """)
                            
                            if 'predictions' not in st.session_state:
                                st.session_state.predictions = {}
                            st.session_state.predictions[patient_id] = {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                'risk': risk
                            }
                        else:
                            st.error(f"Prediction error: {result.get('error', 'Unknown error')}")
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"""
                    Connection error:
                    {str(e)}
                    
                    Please ensure:
                    1. The prediction server is running
                    2. The endpoint URL is correct
                    3. There are no network issues
                    """)
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
    
    # ===== SIDEBAR CONTENT =====
    st.sidebar.header("Clinical Overview")
    
    # Notes summary
    st.sidebar.subheader("Notes Preview")
    if st.session_state.clinical_notes[patient_id]['last_updated']:
        st.sidebar.write(f"Last updated: {st.session_state.clinical_notes[patient_id]['last_updated']}")
    
    st.sidebar.write("**Doctor Notes:**")
    st.sidebar.text(st.session_state.clinical_notes[patient_id]['doctor_notes'][:200] + 
                   ("..." if len(st.session_state.clinical_notes[patient_id]['doctor_notes']) > 200 else ""))
    
    st.sidebar.write("**Nurse Notes:**")
    st.sidebar.text(st.session_state.clinical_notes[patient_id]['nurse_notes'][:200] + 
                   ("..." if len(st.session_state.clinical_notes[patient_id]['nurse_notes']) > 200 else ""))
    
    # Clinical team assignment
    st.sidebar.subheader("Clinical Team")
    if 'clinical_assignments' not in st.session_state:
        st.session_state.clinical_assignments = {}
    
    if patient_id not in st.session_state.clinical_assignments:
        st.session_state.clinical_assignments[patient_id] = {
            'doctor': f"Dr. {fake.last_name()}",
            'nurse': f"Nurse {fake.last_name()}",
            'notes': f"Initial assessment: {patient_info['condition']}"
        }
    
    with st.sidebar.form("team_assignment", clear_on_submit=True):  # Add clear_on_submit=True
        st.session_state.clinical_assignments[patient_id]['doctor'] = st.text_input(
            "Attending Physician",
            value=st.session_state.clinical_assignments[patient_id]['doctor']
        )
        
        st.session_state.clinical_assignments[patient_id]['nurse'] = st.text_input(
            "Primary Nurse",
            value=st.session_state.clinical_assignments[patient_id]['nurse']
        )
        
        if st.form_submit_button("Save Team"):
            st.sidebar.success("Team assignments updated!")
    
    st.sidebar.write(f"**Physician:** {st.session_state.clinical_assignments[patient_id]['doctor']}")
    st.sidebar.write(f"**Nurse:** {st.session_state.clinical_assignments[patient_id]['nurse']}")
    
    # Chatbot integration
    st.sidebar.header("ICU Assistant")
    with st.sidebar.form("chatbot_form", clear_on_submit=True):  # Wrap in form with clear_on_submit
        user_query = st.text_input("Ask about patient status", key="chatbot_query")
        submitted = st.form_submit_button("Ask")
        
        if submitted and user_query:
            response = requests.post(
                'http://localhost:5001/chatbot',
                json={'query': user_query, 'patient_id': patient_id},
                timeout=20
            )
            if response.status_code == 200:
                st.sidebar.success(f"Assistant: {response.json().get('response', '')}")
            else:
                st.sidebar.error("Chatbot unavailable")

if __name__ == "__main__":
    main()