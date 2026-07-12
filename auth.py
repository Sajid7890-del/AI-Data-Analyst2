import hashlib
import os
import streamlit as st
import database

def generate_salt():
    """Generate a random 16-byte salt as a hex string."""
    return os.urandom(16).hex()

def hash_password(password, salt):
    """Hash the password with the salt using PBKDF2 with SHA-256."""
    password_bytes = password.encode('utf-8')
    salt_bytes = bytes.fromhex(salt)
    
    # Secure key derivation with 100,000 iterations
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password_bytes,
        salt_bytes,
        100000
    )
    return key.hex()

def register_user(username, password):
    """Register a new user in the database. Returns (success, message)."""
    username = username.strip()
    password = password.strip()
    
    if not username or not password:
        return False, "Username and password cannot be empty."
        
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
        
    # Check if user already exists
    existing = database.get_user_by_username(username)
    if existing:
        return False, f"Username '{username}' is already taken."
        
    # Hash password & save
    salt = generate_salt()
    pw_hash = hash_password(password, salt)
    
    success = database.create_user(username, pw_hash, salt)
    if success:
        return True, "Registration successful! You can now log in."
    else:
        return False, "An error occurred during registration. Please try again."

def authenticate_user(username, password):
    """Verify user credentials. Returns the user row (as dict) or None if verification fails."""
    username = username.strip()
    password = password.strip()
    
    user = database.get_user_by_username(username)
    if not user:
        return None
        
    db_hash = user['password_hash']
    db_salt = user['salt']
    
    computed_hash = hash_password(password, db_salt)
    if computed_hash == db_hash:
        return dict(user)
    return None

def init_auth_session():
    """Ensure user session state variables exist."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "current_dataset_id" not in st.session_state:
        st.session_state.current_dataset_id = None
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "provider" not in st.session_state:
        st.session_state.provider = "Mock Engine (No Key Required)"
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "model_name" not in st.session_state:
        st.session_state.model_name = ""
    if "charts_for_pdf" not in st.session_state:
        st.session_state.charts_for_pdf = []

def run_login_flow(username, password):
    """Attempt to log in the user and set session variables."""
    user = authenticate_user(username, password)
    if user:
        st.session_state.logged_in = True
        st.session_state.user = user
        st.session_state.current_dataset_id = None
        st.session_state.current_session_id = None
        return True
    return False

def run_logout():
    """Log out the current user and reset session variables."""
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.current_dataset_id = None
    st.session_state.current_session_id = None
    st.session_state.df = None
    st.session_state.charts_for_pdf = []
    st.session_state.provider = "Mock Engine (No Key Required)"
    st.session_state.api_key = ""
    st.session_state.model_name = ""
    st.rerun()
