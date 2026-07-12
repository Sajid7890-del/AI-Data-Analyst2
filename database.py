import sqlite3
from datetime import datetime
from config import DB_PATH

def get_db_connection():
    """Establish a connection to the SQLite database with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize all tables in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    
    # 2. Datasets Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        col_count INTEGER NOT NULL,
        upload_time TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)
    
    # 3. Chat Sessions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        dataset_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
        FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
    )
    """)
    
    # 4. Chat History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        role TEXT NOT NULL, -- 'user' or 'assistant'
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
    )
    """)
    
    # 5. Analysis History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        dataset_id INTEGER NOT NULL,
        analysis_type TEXT NOT NULL, -- 'Descriptive', 'Outlier', 'Correlation', etc.
        result_summary TEXT NOT NULL,
        executed_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
        FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
    )
    """)
    
    conn.commit()
    conn.close()

# --- User Management ---

def create_user(username, password_hash, salt):
    """Insert a new user record. Returns True if successful, False otherwise."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username.lower().strip(), password_hash, salt, datetime.now().isoformat())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user_by_username(username):
    """Retrieve user details by username."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.lower().strip(),))
    user = cursor.fetchone()
    conn.close()
    return user

# --- Dataset Management ---

def save_dataset_meta(user_id, filename, file_path, row_count, col_count):
    """Save details of an uploaded dataset and return the new row ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO datasets (user_id, filename, file_path, row_count, col_count, upload_time) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, filename, file_path, row_count, col_count, datetime.now().isoformat())
    )
    conn.commit()
    dataset_id = cursor.lastrowid
    conn.close()
    return dataset_id

def get_datasets_by_user(user_id):
    """Retrieve all datasets uploaded by a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE user_id = ? ORDER BY id DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_dataset_by_id(dataset_id):
    """Retrieve a specific dataset by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()
    return row

def delete_dataset(dataset_id):
    """Delete a dataset and its associated record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    conn.commit()
    conn.close()

# --- Chat Session Management ---

def create_chat_session(user_id, dataset_id, name):
    """Create a new chat session for a user-dataset combination."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_sessions (user_id, dataset_id, name, created_at) VALUES (?, ?, ?, ?)",
        (user_id, dataset_id, name, datetime.now().isoformat())
    )
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id

def get_chat_sessions(user_id, dataset_id):
    """Retrieve all chat sessions associated with a user and a dataset."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM chat_sessions WHERE user_id = ? AND dataset_id = ? ORDER BY id DESC",
        (user_id, dataset_id)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_chat_session(session_id):
    """Delete a chat session and all messages in its history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

def save_chat_message(session_id, role, content):
    """Append a message to a chat session's history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    """Retrieve the full messaging log for a chat session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM chat_history WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

# --- Analysis Logs ---

def log_analysis(user_id, dataset_id, analysis_type, result_summary):
    """Record that an analysis operation occurred."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO analysis_history (user_id, dataset_id, analysis_type, result_summary, executed_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, dataset_id, analysis_type, result_summary, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_analysis_history(user_id):
    """Retrieve analysis log history for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ah.*, d.filename FROM analysis_history ah JOIN datasets d ON ah.dataset_id = d.id WHERE ah.user_id = ? ORDER BY ah.id DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows
