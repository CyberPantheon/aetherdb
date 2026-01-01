#!/usr/bin/env python3

"""
Aether DB - Enhanced Streamlit Frontend

A beautiful, intuitive interface for natural language to SQL conversion
with real-time database execution and visualization.

Enhanced with better UX, real-time feedback, and improved integration.
Supports robust database credential handling for various authentication scenarios.
"""
import threading
from contextlib import contextmanager
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import sys
import time # Re-added for explicit time usage in FileDatabaseManager and connect_to_database
import tempfile
import sqlite3
import re
import json
from typing import Optional
from dotenv import load_dotenv
from credential_handler import CredentialHandler, EdgeCaseHandler
import logging

# Configure logging for console feedback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DebugLogger:
    """Custom logger that sends messages to Streamlit session state"""
    def __init__(self):
        self.logs = []
    
    def log(self, message: str, level: str = "INFO"):
        """Add a log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        self.logs.append(log_entry)
        # Print to console as well
        print(f"[{timestamp}] {level}: {message}")
        
    def info(self, message: str):
        self.log(message, "INFO")
    
    def error(self, message: str):
        self.log(message, "ERROR")
    
    def success(self, message: str):
        self.log(message, "SUCCESS")
    
    def warning(self, message: str):
        self.log(message, "WARNING")
    
    def get_logs(self):
        return self.logs
    
    def clear(self):
        self.logs = []

class ThreadSafeSQLiteConnection:
    """Thread-safe SQLite connection manager"""
    _local = threading.local()
    _lock = threading.Lock()
    
    @classmethod
    def get_connection(cls, db_file):
        if not hasattr(cls._local, 'connections'):
            cls._local.connections = {}
        
        if db_file not in cls._local.connections:
            # Removed import sqlite3 as it is imported globally
            cls._local.connections[db_file] = sqlite3.connect(
                db_file, 
                check_same_thread=False
            )
        
        return cls._local.connections[db_file]
    
    @classmethod
    # Used from contextlib import contextmanager, so no need for explicit import at top
    @contextmanager
    def connection(cls, db_file):
        conn = cls.get_connection(db_file)
        try:
            yield conn
        # Replaced bare except: with except Exception:
        except Exception:
            conn.rollback()
            raise
        finally:
            pass

    @classmethod
    def close_connection(cls, db_file):
        """Close a specific connection"""
        if hasattr(cls._local, 'connections') and db_file in cls._local.connections:
            cls._local.connections[db_file].close()
            del cls._local.connections[db_file]
# Add current directory to path to import local modules
sys.path.append(os.path.dirname(__file__))

# Load environment variables from .env
load_dotenv()
# Load environment variables from .env
# load_dotenv() # Removed redundant call

# Get the API key from environment
api_key = os.getenv("GEMINI_API_KEY")

# Safe debug check without exposing sensitive info
if api_key:
    print("Environment configuration loaded successfully ✅")
else:
    print("AI service not configured")
    MODULES_AVAILABLE = True
    GEMINI_API_VALID = False


# Import our modules
try:
    from schema_awareness import SchemaAwarenessModule
    from sqlm import GeminiReasoner, CommandPayload
    from db_executor import DatabaseExecutor
    from data_analyzer import DataAnalyzer, AnalysisSession, TaskStatus
    
    # Test Gemini API securely
    import google.generativeai as genai
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = list(genai.list_models())
            gemini_models = [m for m in models if 'gemini' in m.name]
            print(f"✅ AI service connected successfully! Available models: {len(gemini_models)}")
            MODULES_AVAILABLE = True
            GEMINI_API_VALID = True
        # Replaced 'except Exception as e:' with 'except Exception:' as 'e' was unused
        except Exception:
            print("❌ AI service connection failed: Check API key configuration")
            MODULES_AVAILABLE = True
            GEMINI_API_VALID = False
    else:
        print("❌ AI service not configured")
        MODULES_AVAILABLE = True
        GEMINI_API_VALID = False
        
# Replaced 'except ImportError as e:' with 'except ImportError:' as 'e' was unused
except ImportError:
    print("❌ Module import error")
    MODULES_AVAILABLE = False
    GEMINI_API_VALID = False


# Page configuration with enhanced settings
st.set_page_config(
    page_title="AetherDB - AI-Powered SQL Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/BU-SENG/foss-project-cow-print',
        'Report a bug': "https://github.com/BU-SENG/foss-project-cow-print/issues",
        'About': "# AetherDB\nTransform natural language into SQL with AI!"
    }
)

# Enhanced Custom CSS with modern design
st.markdown("""
<style>
    /* Main background with gradient animation */
    .main {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: transparent;
    }

    /* Enhanced sidebar */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
        border-right: 1px solid #44475a;
    }

    /* Modern cards with glassmorphism effect */
    .glass-card {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
        backdrop-filter: blur(5px);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }

    /* Enhanced buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    /* Enhanced metrics */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.75rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }

    /* Code block styling */
    .stCodeBlock {
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0.5rem 0.5rem 0 0;
        gap: 1rem;
        padding: 0 1rem;
    }
</style>
""", unsafe_allow_html=True)

def sanitize_file_path(file_path: str) -> Optional[str]:
    """
    Sanitize file path to prevent path traversal attacks and other vulnerabilities
    """
    if not file_path or not isinstance(file_path, str):
        return None
    
    normalized_path = os.path.normpath(file_path)
    
    if '..' in normalized_path.split(os.sep):
        print(f"Security: Blocked path traversal attempt: {file_path}")
        return None
    
    try:
        abs_path = os.path.abspath(normalized_path)
        current_dir = os.path.abspath(os.getcwd())
        
        # Merge nested if conditions
        if not abs_path.startswith(current_dir):
            print(f"Security: Blocked path outside working directory: {file_path}")
            return None
        
        allowed_extensions = ['.db', '.sqlite', '.sqlite3', '.db3', '']
        file_ext = os.path.splitext(abs_path)[1].lower()
        if file_ext not in allowed_extensions:
            print(f"Security: Blocked invalid file extension: {file_ext}")
            return None
            
    # Replaced 'except Exception as e:' with 'except Exception:' as 'e' was unused
    except Exception:
        print("Security: Path validation error")
        return None
    
    return abs_path

def is_valid_sqlite_file(file_path: str) -> bool:
    """
    Validate that the file is a legitimate SQLite database file
    """
    try:
        if not os.path.isfile(file_path):
            return False
        
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:
            print("Security: File too large")
            return False
        
        sqlite_header = b'SQLite format 3\x00'
        with open(file_path, 'rb') as f:
            header = f.read(16)
        
        if header == sqlite_header:
            return True
        
        print("Security: Invalid SQLite file header")
        return False
            
    # Replaced 'except Exception as e:' with 'except Exception:' as 'e' was unused
    except Exception:
        print("Security: SQLite file validation error")
        return False

def display_enhanced_header():
    """Display modern app header with import status"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: white; font-size: 3.5rem; margin-bottom: 0.5rem; font-weight: 700;'>
                AetherDB
            </h1>
            <p style='color: rgba(255,255,255,0.9); font-size: 1.3rem; margin-bottom: 1rem;'>
                AI-Powered Natural Language to SQL
            </p>
            <div style='display: inline-flex; gap: 0.5rem; background: rgba(255,255,255,0.1); 
                        padding: 0.5rem 1rem; border-radius: 2rem; backdrop-filter: blur(10px);'>
                <span style='color: #10b981;'>●</span>
                <span style='color: white;'>Import SQL files or connect to databases</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def initialize_reasoner(schema_text: str) -> Optional[GeminiReasoner]:
    """Initialize the Gemini Reasoner with proper error handling"""
    try:
        return GeminiReasoner(schema_snapshot=schema_text, api_key=api_key)
    # Replaced 'except Exception as e:' with 'except Exception as _:' as 'e' was unused
    except Exception as _:
        print(f"Error initializing reasoner: {_}")
        return None

def sidebar_database_connection():
    """Clean sidebar with single SQL import option"""
    st.sidebar.title("AetherDB")
    
    if not GEMINI_API_VALID:
        st.sidebar.warning("AI Mode: Demo (Check API Key)")
        with st.sidebar.expander("API Key Help"):
            st.markdown("""
            **To enable full AI features:**
            1. Get API key from [Google AI Studio](https://aistudio.google.com/)
            2. Add to `.env` file:
            ```
            GEMINI_API_KEY=your_actual_key_here
            ```
            3. Restart the app
            """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Start")
    
    sql_file = st.sidebar.file_uploader(
        "Upload SQL File", 
        type=['sql', 'txt'],
        help="Upload SQL file to create instant queryable database",
        key="main_sql_import"
    )
    
    if sql_file is not None and st.sidebar.button("Create Database & Start Querying", 
                                                   use_container_width=True, 
                                                   type="primary",
                                                   key="create_db_from_sql"):
        execute_uploaded_sql_file(sql_file, 'sqlite')
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Database Connection")
    
    if not st.session_state.connected:
        with st.sidebar.expander("Connect to Existing Database", expanded=False):
            db_type = st.selectbox(
                "Database Type",
                ["MySQL", "PostgreSQL", "SQLite"],
                help="Select your database type"
            )
            
            if db_type == "SQLite":
                db_file = st.text_input(
                    "Database File Path",
                    value="database.db",
                    help="Path to your SQLite database file"
                )
                
                if connect_btn := st.button(
                    "Connect to SQLite", 
                    use_container_width=True,
                    type="secondary"
                ):
                    sanitized_path = sanitize_file_path(db_file)
                    if sanitized_path and os.path.exists(sanitized_path):
                        if is_valid_sqlite_file(sanitized_path):
                            connect_to_database(db_type.lower(), database=sanitized_path)
                        else:
                            st.sidebar.error("Invalid SQLite database file")
                    else:
                        st.sidebar.error("Invalid database file path or file not found")
            
            else:
                col1, col2 = st.columns(2)
                with col1:
                    host = st.text_input("Host", value="localhost")
                    port = st.number_input(
                        "Port", 
                        value=3306 if db_type == "MySQL" else 5432,
                        min_value=1,
                        max_value=65535
                    )
                with col2:
                    user = st.text_input("Username", placeholder="Enter username")
                    password = st.text_input("Password", type="password", placeholder="Leave blank if not required")
                
                database = st.text_input("Database Name", placeholder="Enter database name")
                
                # Informational note about optional password
                st.caption("💡 Password is optional - leave blank if your database doesn't require authentication")
                
                connect_btn = st.button(
                    f"Connect to {db_type}", 
                    use_container_width=True,
                    type="secondary"
                )
                
                if connect_btn:
                    if not all([user, database]):
                        st.sidebar.error("Username and database name are required")
                    else:
                        conn_params = {
                            "host": host,
                            "port": port,
                            "user": user,
                            "database": database
                        }
                        # Only include password if provided (handles edge cases where no password is needed)
                        if password:
                            conn_params["password"] = password
                        else:
                            conn_params["password"] = ""
                        
                        connect_to_database(db_type.lower(), **conn_params)
    
    else:
        st.sidebar.success("Database Connected")
        
        db_name = "Unknown Database"
        db_type = "Unknown"
        table_count = 0
        
        if st.session_state.working_with_file_db and st.session_state.current_file_db:
            db_info = st.session_state.file_db_manager.get_current_db_info()
            if db_info:
                db_name = db_info['name']
                db_type = "SQLITE"
                table_count = len(db_info['tables'])
        
        elif (st.session_state.last_connection_type == 'sqlite' and 
              st.session_state.sam and 
              hasattr(st.session_state.sam, 'database_file')):
            db_path = st.session_state.sam.database_file
            db_name = os.path.basename(db_path)
            db_type = "SQLITE"
            try:
                conn = sqlite3.connect(db_path, check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                table_count = len(tables)
                conn.close()
            # Replaced 'except Exception:' with 'except Exception:'
            except Exception:
                table_count = 0
        
        elif (st.session_state.sam and 
              hasattr(st.session_state.sam, 'current_metadata')):
            metadata = st.session_state.sam.current_metadata
            db_name = getattr(metadata, 'database_name', 'Connected Database')
            db_type = getattr(metadata, 'database_type', 'Unknown').upper()
            table_count = getattr(metadata, 'table_count', 0)
        
        st.sidebar.markdown("""
        <div style='
            background: rgba(255,255,255,0.1); 
            padding: 1.5rem; 
            border-radius: 0.75rem;
            border: 1px solid rgba(255,255,255,0.2);
            margin: 1rem 0;
        '>
        """, unsafe_allow_html=True)
        
        info_cols = st.sidebar.columns(2)
        
        with info_cols[0]:
            st.metric(
                label="Database", 
                value=db_name,
                help="Currently connected database"
            )
            st.metric(
                label="Tables", 
                value=table_count,
                help="Number of tables in database"
            )
        
        with info_cols[1]:
            st.metric(
                label="Type", 
                value=db_type,
                help="Database type"
            )
            st.metric(
                label="Status", 
                value="Active",
                help="Connection status"
            )
        
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
        
        if st.session_state.working_with_file_db and st.session_state.current_file_db:
            db_info = st.session_state.file_db_manager.get_current_db_info()
            if db_info:
                st.sidebar.markdown("---")
                st.sidebar.markdown("### Imported Database")
                st.sidebar.info(f"**{db_info['name']}**")
                st.sidebar.write(f"Tables: {len(db_info['tables'])}")
                st.sidebar.write(f"Created: {db_info['created_at'].strftime('%H:%M:%S')}")
                
                if st.sidebar.button("Switch to Main DB", use_container_width=True, key="switch_to_main_db"):
                    switch_to_main_database()
        
        st.sidebar.markdown("### Management")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Refresh Schema", use_container_width=True, key="refresh_schema_btn"):
                refresh_schema()
        with col2:
            if st.button("Force Table Detect", use_container_width=True, key="force_table_detect"):
                if st.session_state.reasoner and hasattr(st.session_state.reasoner, 'get_actual_tables'):
                    st.session_state.reasoner.table_names = st.session_state.reasoner.get_actual_tables()
                    st.sidebar.success("Table detection forced!")
                st.rerun()
        
        if st.sidebar.button("Disconnect", use_container_width=True, type="secondary", key="disconnect_db"):
            disconnect_database()
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("Example Queries"):
        st.markdown("""
        **Try asking:**
        - "Show me all users"
        - "Count orders by status"  
        - "Find recent transactions"
        - "Average sales by region"
        - "List products with low stock"
        """)

def refresh_schema():
    """Refresh the database schema"""
    if st.session_state.sam:
        try:
            st.session_state.sam.generate_full_schema()
            st.sidebar.success("Schema refreshed successfully!")
            st.rerun()
        # Replaced 'except Exception as e:' with 'except Exception as _:' as 'e' was unused
        except Exception as _:
            st.sidebar.error(f"Failed to refresh schema: {_}")

def generate_visualizations(df: pd.DataFrame):
    """
    Dynamically generate visualization options based on the dataframe content.
    """
    if df.empty or len(df.columns) < 2:
        return

    st.markdown("---")
    st.header("🎨 Visualizations")

    # Identify column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    all_cols = df.columns.tolist()

    # Smart Defaults
    default_x = all_cols[0]
    default_y = numeric_cols[0] if numeric_cols else all_cols[-1]
    
    # Visualization Controls
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            chart_type = st.selectbox(
                "Chart Type",
                ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Area Chart", "Histogram"],
                index=0
            )
        
        with col2:
            x_axis = st.selectbox("X Axis", all_cols, index=all_cols.index(default_x) if default_x in all_cols else 0)
            
        with col3:
            # Pie charts don't strictly need a Y axis (can count), but usually map a value
            if chart_type != "Histogram":
                y_choices = numeric_cols or all_cols
                default_y_index = y_choices.index(default_y) if default_y in y_choices else 0
                y_axis = st.selectbox("Y Axis (Value/Metric)", y_choices, index=default_y_index)
            else:
                y_axis = None

        # Color/Group by option
        use_color = st.checkbox("Group by Color (Legend)", value=False)
        if use_color:
            color_col = st.selectbox("Select Grouping Column", categorical_cols or all_cols)
        else:
            color_col = None

    # Generate Plotly Charts
    try:
        fig = None
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} by {x_axis}", template="plotly_dark")
        
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis, color=color_col, markers=True, title=f"{y_axis} over {x_axis}", template="plotly_dark")
        
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} vs {x_axis}", template="plotly_dark")
        
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_axis, values=y_axis, title=f"Distribution of {y_axis} by {x_axis}", template="plotly_dark")
            
        elif chart_type == "Area Chart":
            fig = px.area(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} by {x_axis}", template="plotly_dark")

        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis, color=color_col, title=f"Distribution of {x_axis}", template="plotly_dark")

        if fig:
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not generate {chart_type} with selected data. Try changing the axes. (Error: {str(e)})")
        
def switch_to_main_database():
    """Switch back to main database from file database"""
    if st.session_state.working_with_file_db and st.session_state.current_file_db:
        db_name = st.session_state.current_file_db
        if hasattr(st.session_state, 'file_db_manager'):
            st.session_state.file_db_manager.cleanup_temp_database(db_name)
        
        st.session_state.working_with_file_db = False
        st.session_state.current_file_db = None
        
        st.sidebar.success("Switched back to main database")
        st.rerun()

class FileDatabaseManager:
    """Manages temporary databases created from SQL files"""
    
    def __init__(self):
        self.temp_databases = {}
        self.current_file_db = None
    
    def create_temp_database_from_sql(self, sql_content: str, db_name: str = None):
        """Create a temporary SQLite database from SQL file content"""
        # Used named expression to simplify assignment and conditional
        if not (db_name := db_name or f"imported_db_{int(time.time())}"):
            db_name = f"imported_db_{int(time.time())}"
        
        db_name = re.sub(r'[^a-zA-Z0-9_-]', '', db_name) or "imported_db"
        
        temp_dir = tempfile.gettempdir()
        temp_db_path = os.path.join(temp_dir, f"{db_name}.db")
        
        temp_db_path = sanitize_file_path(temp_db_path)
        if not temp_db_path:
            print("Security: Invalid temporary database path")
            return None
        
        try:
            conn = sqlite3.connect(temp_db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            executed_count = 0
            for statement in statements:
                if statement and statement.strip():
                    try:
                        cursor.execute(statement)
                        executed_count += 1
                    # Replaced 'except Exception as e:' with 'except Exception as _:' as 'e' was unused
                    except Exception as _:
                        if "no such table" not in str(_).lower():
                            print("Info: Could not execute statement: Schema mismatch")
            
            conn.commit()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            self.temp_databases[db_name] = {
                'path': temp_db_path,
                'name': db_name,
                'created_at': datetime.now(),
                'tables': tables,
                'statements_executed': executed_count,
                'total_statements': len(statements)
            }
            
            self.current_file_db = db_name
            return db_name
            
        # Replaced 'except Exception:' with 'except Exception as _:'
        except Exception as _:
            print("Error creating temp database: Configuration error")
            if os.path.exists(temp_db_path):
                from contextlib import suppress
                with suppress(Exception):
                    os.remove(temp_db_path)
            return None
    
    def get_current_db_info(self):
        """Get info about current file database"""
        if self.current_file_db and self.current_file_db in self.temp_databases:
            return self.temp_databases[self.current_file_db]
        return None

    def cleanup_temp_database(self, db_name: str):
        """Clean up a temporary database file"""
        if db_name not in self.temp_databases:
            return
        
        db_info = self.temp_databases[db_name]
        try:
            ThreadSafeSQLiteConnection.close_connection(db_info['path'])
            if os.path.exists(db_info['path']):
                os.remove(db_info['path'])
        # Replaced 'except Exception:' with 'except Exception:'
        except Exception:
            print("Warning: Could not clean up temp database")
        
        del self.temp_databases[db_name]
        
        if self.current_file_db == db_name:
            self.current_file_db = None

    def cleanup_all(self):
        """Clean up all temporary databases"""
        for db_name in list(self.temp_databases.keys()):
            self.cleanup_temp_database(db_name)

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'connected': False,
        'sam': None,
        'reasoner': None,
        'executor': None,
        'query_history': [],
        'current_schema': 'all',
        'selected_tables': [],
        'last_connection_type': None,
        'ai_thinking': False,
        'execution_stats': {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_execution_time': 0,
            'average_confidence': 0
        },
        'user_preferences': {
            'auto_refresh_schema': True,
            'show_explanations': True,
            'enable_visualizations': True,
            'theme': 'dark'
        },
        'current_results': None,
        'active_tab': 'query',
        'file_db_manager': FileDatabaseManager(),
        'working_with_file_db': False,
        'current_file_db': None,
        # --- NEW STATE VARIABLES ---
        'generated_sql': "",
        'query_results': None,
        'last_nl_query': "",
        'debug_logger': DebugLogger(),
        'show_debug_console': False,
        # --- ANALYSIS MODE VARIABLES ---
        'analysis_mode': False,
        'data_analyzer': None,
        'current_analysis_session': None,
        'analysis_history': [],
        'selected_analysis_tables': [],
        'current_analysis_query': ""
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def execute_uploaded_sql_file(sql_file, db_type, **conn_params):
    """Execute SQL from uploaded file and create queryable database"""
    try:
        sql_content = sql_file.getvalue().decode("utf-8")
        
        db_name = f"imported_{int(time.time())}"
        # Used named expression to simplify assignment and conditional
        if db_id := st.session_state.file_db_manager.create_temp_database_from_sql(sql_content, db_name):
            db_info = st.session_state.file_db_manager.get_current_db_info()
            
            sanitized_path = sanitize_file_path(db_info['path'])
            if not sanitized_path:
                st.sidebar.error("Security: Invalid database path")
                return
                
            if connect_to_database('sqlite', database=sanitized_path):
                st.session_state.working_with_file_db = True
                st.session_state.current_file_db = db_id
                
                st.sidebar.success("Database created from SQL file!")
                st.sidebar.info(f"Found {len(db_info['tables'])} tables")
                
                refresh_schema()
                
                if st.session_state.sam and hasattr(st.session_state.sam, 'schema_file'):
                    try:
                        with open(st.session_state.sam.schema_file, 'r') as f:
                            schema_text = f.read()
                        if st.session_state.reasoner and hasattr(st.session_state.reasoner, 'update_schema'):
                            st.session_state.reasoner.update_schema(schema_text)
                    # Replaced 'except Exception as e:' with 'except Exception:' as 'e' was unused
                    except Exception:
                        print("Warning: Could not update reasoner schema")
                
                st.session_state.active_tab = 'query'
                st.rerun()
            else:
                st.sidebar.error("Failed to connect to created database")
        else:
            st.sidebar.error("Failed to create database from SQL file")
                
    # Replaced 'except Exception as e:' with 'except Exception:' as 'e' was unused
    except Exception:
        st.sidebar.error("Failed to process SQL file")

def connect_to_database(db_type: str, **conn_params):
    """Database connection handler with proper thread handling and path sanitization"""
    with st.spinner("🔌 Connecting to database..."):
        try:
            if not MODULES_AVAILABLE:
                st.sidebar.error("❌ Required modules not available")
                return False
            
            sam = SchemaAwarenessModule()
            
            if db_type == 'sqlite' and 'database' in conn_params:
                db_file = conn_params['database']
                
                if not os.path.exists(db_file):
                    st.sidebar.error(f"❌ Database file not found: {os.path.basename(db_file)}")
                    return False
            
            st.sidebar.info("📡 Connecting to database...")
            
            if sam.connect_database(db_type, **conn_params):
                st.session_state.sam = sam
                st.session_state.connected = True
                st.session_state.last_connection_type = db_type
                
                try:
                    with open(sam.schema_file, 'r') as f:
                        schema_text = f.read()
                    
                    st.sidebar.info("🤖 Initializing AI components...")
                    st.session_state.reasoner = initialize_reasoner(schema_text)
                    
                    st.session_state.executor = DatabaseExecutor(sam.connection, db_type)
                    
                    st.sidebar.success("✅ Database connected successfully!")
                    st.sidebar.success("✅ AI components initialized!")
                    return True
                    
                # Replaced 'except Exception as e:' with 'except Exception as _:' as 'e' was unused
                except Exception as _:
                    st.sidebar.error("⚠️ Error initializing AI components")
                    st.session_state.connected = True
                    st.sidebar.success("✅ Database connected (AI features limited)")
                    return True
            else:
                st.sidebar.error("❌ Database connection failed")
                return False
                
        # Replaced 'except Exception as e:' with 'except Exception as _:' as 'e' was unused
        except Exception as _:
            st.sidebar.error(f"❌ Connection error: {str(_)}")
            return False
                 
def disconnect_database():
    """Enhanced database disconnection"""
    try:
        if st.session_state.connected:
            if st.session_state.working_with_file_db and st.session_state.current_file_db:
                db_name = st.session_state.current_file_db
                if hasattr(st.session_state, 'file_db_manager'):
                    st.session_state.file_db_manager.cleanup_temp_database(db_name)
            
            if st.session_state.sam:
                try:
                    st.session_state.sam.close()
                # Replaced 'except Exception as e:' with 'except Exception:' as 'e' was unused
                except Exception:
                    print("Warning: Error closing SAM")
            
            st.session_state.connected = False
            st.session_state.sam = None
            st.session_state.reasoner = None
            st.session_state.executor = None
            st.session_state.current_results = None
            st.session_state.working_with_file_db = False
            st.session_state.current_file_db = None
            st.session_state.query_results = None # Clean up results
            
            st.sidebar.success("Disconnected from database")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.info("Not connected to any database")
            
    # Replaced 'except Exception as e:' with 'except Exception:' as 'e' was unused
    except Exception:
        st.sidebar.error("Error during disconnection")

def display_sql_editor_and_execution():
    """Display SQL editor with refinement and execution controls"""
    if st.session_state.generated_sql:
        st.subheader("📝 SQL Editor")
        
        # Refinement Section (Collapsible)
        with st.expander("✨ AI Refinement (Optional)"):
            refine_instruction = st.text_input("How should the SQL be modified?", placeholder="e.g., Filter by date > 2023")
            if st.button("Refine Code", key="refine_code_btn") and st.session_state.generated_sql and refine_instruction:
                with st.spinner("🤖 Refining via Gemini API..."):
                    try:
                        logger = st.session_state.get('debug_logger')
                        if logger:
                            logger.info(f"Refinement request: {refine_instruction}")
                            logger.info("Sending refinement request to Gemini API...")
                        
                        st.info("📡 Sending refinement request to Gemini API...")
                        
                        # Heuristic prompt construction for refinement
                        refine_prompt = f"Existing SQL: {st.session_state.generated_sql}\n\nModification request: {refine_instruction}\n\nReturn only the updated SQL."
                        payload = CommandPayload(
                            intent="query",
                            raw_nl=refine_prompt,
                            dialect=st.session_state.last_connection_type or "sqlite"
                        )
                        output = st.session_state.reasoner.generate(payload)
                        
                        st.success("✅ Refinement received from API!")
                        if logger:
                            logger.success("Refinement completed successfully")
                        
                        if output and hasattr(output, 'thought_process') and output.thought_process:
                            st.info(f"💭 Refinement Analysis: {output.thought_process}")
                            if logger:
                                logger.info(f"Refinement analysis: {output.thought_process[:100]}...")
                        
                        st.session_state.generated_sql = output.sql
                        if logger:
                            logger.info(f"Refined SQL: {output.sql[:100]}...")
                        # Don't rerun - the editor will update with the refined SQL
                    except Exception as e:
                        error_msg = f"Refinement failed: {str(e)}"
                        st.error(f"❌ {error_msg}")
                        logger = st.session_state.get('debug_logger')
                        if logger:
                            logger.error(error_msg)
                        st.info("💡 Tip: Check your API key and internet connection")

        # The SQL Editor
        # We use a key to sync this widget with session state, but we also want to allow manual edits.
        sql_query = st.text_area(
            "SQL Query", 
            value=st.session_state.generated_sql, 
            height=150,
            key="sql_editor_widget"
        )
        
        # Sync manual edits back to session state
        st.session_state.generated_sql = sql_query

        # Execution
        col_exec1, col_exec2, col_exec3 = st.columns([1, 1, 3])
        with col_exec1:
            execute_btn = st.button("🚀 Run Query", type="primary", use_container_width=True, key="run_query_btn")
        with col_exec2:
            allow_destructive = st.checkbox("Allow Changes", value=False, help="Enable INSERT/UPDATE/DELETE")

        if execute_btn:
            with st.spinner("⚙️ Executing query..."):
                try:
                    logger = st.session_state.get('debug_logger')
                    is_destructive = not sql_query.strip().lower().startswith("select")
                    
                    if logger:
                        logger.info(f"Executing SQL query: {sql_query[:100]}...")
                    
                    if is_destructive and not allow_destructive:
                        msg = "Destructive operations (INSERT/UPDATE/DELETE) disabled - Enable 'Allow Changes' to proceed"
                        st.warning(f"⚠️ {msg}")
                        if logger:
                            logger.warning(msg)
                    else:
                        st.info("📡 Executing query on database...")
                        if logger:
                            logger.info("📡 Executing query on database...")
                        
                        result = st.session_state.executor.execute_query(
                            sql_query,
                            safe_to_execute=True,
                            is_destructive=is_destructive and allow_destructive
                        )
                        st.success("✅ Query executed successfully!")
                        if logger:
                            logger.success(f"Query executed successfully. Rows affected: {len(result) if isinstance(result, list) else 1}")
                        
                        st.session_state.query_results = st.session_state.executor.format_results_for_display(result)
                except Exception as e:
                    error_msg = f"Execution error: {str(e)}"
                    st.error(f"❌ {error_msg}")
                    logger = st.session_state.get('debug_logger')
                    if logger:
                        logger.error(error_msg)
                    
                    with st.expander("📋 Error Details"):
                        import traceback
                        st.code(traceback.format_exc())

def display_query_results():
    """Display query results and visualization"""
    if st.session_state.query_results:
        formatted = st.session_state.query_results
        
        st.markdown("---")
        st.subheader("📊 Results")
        
        if formatted['success']:
            st.success(formatted['message'])
            if formatted['has_data']:
                df = pd.DataFrame(formatted['data'])
                st.dataframe(df, use_container_width=True)
                
                # Visualization (Updates here won't lose state because query_results is in session_state)
                generate_visualizations(df)
                
                # Download
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")
        else:
            st.error(formatted['message'])
            if formatted.get('error'):
                st.code(formatted['error'])

def display_query_interface():
    """Display the main query interface"""
    st.header("🔍 Natural Language Query")
    
    # Initialize session state for this page if not present
    if "generated_sql" not in st.session_state:
        st.session_state.generated_sql = ""
    if "query_results" not in st.session_state:
        st.session_state.query_results = None
    if "last_nl_query" not in st.session_state:
        st.session_state.last_nl_query = ""

    # 1. NL Input
    nl_query = st.text_area(
        "Enter your question:", 
        value=st.session_state.last_nl_query,
        placeholder="e.g., Show me top 10 users...", 
        height=70
    )
    
    col_gen1, col_gen2 = st.columns([1, 4])
    with col_gen1:
        generate_btn = st.button("🤖 Generate SQL", type="primary", use_container_width=True, key="generate_sql_btn")
    
    # Logic: Generate SQL
    if generate_btn and nl_query:
        st.session_state.last_nl_query = nl_query
        
        logger = st.session_state.debug_logger
        logger.info(f"User query: {nl_query}")
        
        with st.spinner("🤖 Generating SQL via Gemini API..."):
            try:
                # Show API request status
                st.info("📡 Sending request to Gemini API...")
                logger.info("Sending request to Gemini API...")
                
                payload = CommandPayload(
                    intent="select",
                    raw_nl=nl_query,
                    dialect=st.session_state.last_connection_type or "sqlite",
                )
                
                logger.info(f"Payload created - Dialect: {st.session_state.last_connection_type or 'sqlite'}")
                
                # Call the reasoner and get response
                output = st.session_state.reasoner.generate(payload)
                
                # Show API response confirmation
                st.success("✅ API Response received!")
                logger.success("API Response received from Gemini!")
                
                # Log the full response for debugging
                logger.info(f"Full API response - Intent: {output.intent}, Confidence: {output.confidence}, Safe: {output.safe_to_execute}")
                if output.errors:
                    logger.error(f"API Errors: {', '.join(output.errors)}")
                if output.warnings:
                    logger.warning(f"API Warnings: {', '.join(output.warnings)}")
                
                # Display AI's thought process if available
                if output and hasattr(output, 'thought_process') and output.thought_process:
                    st.info(f"💭 API Analysis: {output.thought_process}")
                    logger.info(f"Thought process: {output.thought_process}")
                
                if output and hasattr(output, 'sql') and output.sql:
                    st.session_state.generated_sql = output.sql
                    logger.success(f"SQL Generated: {output.sql[:100]}...")
                else:
                    # Log detailed information about why SQL wasn't generated
                    st.session_state.generated_sql = "-- No SQL generated"
                    logger.error(f"No SQL in response - Intent: {output.intent}, Clarify Required: {getattr(output, 'clarify_required', 'N/A')}")
                    logger.warning("No SQL was generated by the API - check intent and explanation below")
                    
                    # Show explanation if available
                    if hasattr(output, 'explain_text') and output.explain_text:
                        st.warning(f"API Response: {output.explain_text}")
                        logger.info(f"API Explanation: {output.explain_text}")
                
                # Clear previous results when new SQL is generated
                st.session_state.query_results = None 
                # Don't rerun - let the page continue to display the SQL editor below
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Generation failed: {error_msg}")
                logger.error(f"Generation failed: {error_msg}")
                st.info("💡 Tip: Make sure your API key is valid and you have internet connection")
                import traceback
                with st.expander("📋 Error Details"):
                    st.code(traceback.format_exc())

    # 2. SQL Editor & Refinement
    display_sql_editor_and_execution()

    # 3. Results & Visualization (Persistent)
    display_query_results()

def display_schema_explorer():
    """Display schema exploration interface"""
    st.header("📋 Database Schema")
    
    if st.session_state.sam:
        tables = st.session_state.sam.get_tables()
        
        if tables:
            st.success(f"Found {len(tables)} tables")
            
            for table in tables:
                with st.expander(f"📊 Table: {table}"):
                    try:
                        schema = st.session_state.sam._get_table_schema(table)
                        
                        st.markdown("**Columns:**")
                        col_data = [
                            {
                                "Name": col['name'],
                                "Type": col['type'],
                                "Nullable": "Yes" if col['nullable'] else "No",
                                "Default": col.get('default', 'None')
                            }
                            for col in schema.columns
                        ]
                        
                        if col_data:
                            st.table(pd.DataFrame(col_data))
                        
                        if schema.primary_keys:
                            st.markdown("**Primary Keys:** " + ", ".join(schema.primary_keys))
                        
                        if schema.foreign_keys:
                            st.markdown("**Foreign Keys:**")
                            for fk in schema.foreign_keys:
                                # Local variable 'fk' is assigned to but never used
                                st.write(f"- {fk['column']} → {fk['references_table']}.{fk['references_column']}")
                        
                        if schema.row_count is not None:
                            st.metric("Row Count", schema.row_count)
                    
                    # Replaced 'except Exception as e:' with 'except Exception as _:' as 'e' was unused
                    except Exception as _:
                        st.error(f"Error loading schema: {_}")
        else:
            st.warning("No tables found in database")
    else:
        st.info("Connect to a database to view schema")

def display_query_history():
    """Display query execution history"""
    st.header("📜 Query History")
    
    if st.session_state.executor:
        history = st.session_state.executor.get_execution_history(limit=20)
        
        if history:
            for i, entry in enumerate(reversed(history)):
                with st.expander(f"Query {len(history) - i} - {entry['status']} - {entry['timestamp']}"):
                    # --- FIX START --- 
                    # Use .get() to handle missing key and format the ms value
                    exec_time_ms = entry.get('execution_time_ms', 0)
                    st.markdown(f"**Execution Time:** {exec_time_ms:.2f}ms")
                    # --- FIX END ---
                    
                    st.markdown("**Rows Affected:** " + str(entry['rows_affected']))
                    
                    if entry.get('error'):
                        st.error(entry['error'])
                    
                    if entry.get('warnings'):
                        st.warning(", ".join(entry['warnings']))
        else:
            st.info("No query history yet")
    else:
        st.info("Connect to a database to view history")
        
def display_query_performance_charts(history):
    """Display query performance charts from execution history"""
    try:
        df = pd.DataFrame(history)
        
        if 'execution_time_ms' in df.columns:
            fig = px.line(
                df,
                x=df.index,
                y='execution_time_ms',
                title='Query Execution Time Trend',
                labels={'execution_time_ms': 'Time (ms)', 'index': 'Query Number'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if 'status' in df.columns:
            status_counts = df['status'].value_counts()
            fig2 = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Query Status Distribution'
            )
            st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        st.error("Error displaying performance charts")

def display_analytics_dashboard():
    """Display analytics dashboard with performance metrics"""
    st.header("📈 Analytics Dashboard")
    
    if st.session_state.executor:
        # --- FIX START ---
        # We use a large number (10000) instead of None to avoid the TypeError
        history = st.session_state.executor.get_execution_history(limit=10000)
        
        total_executions = len(history)
        
        # Count blocked queries
        blocked = len([entry for entry in history if entry.get('status') == 'blocked'])
        
        stats = {
            'total_executions': total_executions,
            'blocked': blocked
        }
        # --- FIX END ---
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Queries", stats.get('total_executions', 0))
        
        with col2:
            st.metric("Blocked Queries", stats.get('blocked', 0))
        
        if stats.get('total_executions', 0) > 0:
            st.markdown("### 📊 Query Performance")
            
            if history:
                display_query_performance_charts(history)
    else:
        st.info("Connect to a database to view analytics")
        
def display_welcome_screen():
    """Display welcome screen when not connected"""
    st.markdown("""
    <div style='text-align: center; padding: 3rem 0;'>
        <h2 style='color: white;'>Welcome to AetherDB! 🚀</h2>
        <p style='color: rgba(255,255,255,0.8); font-size: 1.2rem;'>
            Get started by connecting to a database or uploading a SQL file
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### Quick Start Guide
        
        1. **Upload SQL File** 📁
           - Click "Upload SQL File" in the sidebar
           - Select your .sql file
           - Click "Create Database & Start Querying"
        
        2. **Connect to Database** 🔌
           - Expand "Connect to Existing Database"
           - Choose your database type
           - Enter connection details
           - Click "Connect"
        
        3. **Start Querying** 💬
           - Type natural language questions
           - AI will generate SQL
           - Execute and view results
        
        ### Features
        
        ✅ Natural language to SQL translation  
        ✅ Support for MySQL, PostgreSQL, SQLite  
        ✅ Schema exploration and visualization  
        ✅ Query history and analytics  
        ✅ Safe execution with validation  
        """)

def debug_database_status():
    """Debug function to show current database connection status"""
    if st.session_state.get('connected'):
        with st.expander("🔍 Debug: Database Status", expanded=False):
            st.write("**Connection Status:**", "✅ Connected")
            st.write("**DB Type:**", st.session_state.get('last_connection_type', 'Unknown'))
            st.write("**SAM Available:**", st.session_state.sam is not None)
            st.write("**Reasoner Available:**", st.session_state.reasoner is not None)
            st.write("**Executor Available:**", st.session_state.executor is not None)
            st.write("**Working with File DB:**", st.session_state.get('working_with_file_db', False))
            
            if st.session_state.sam:
                tables = st.session_state.sam.get_tables()
                st.write("**Tables Found:**", len(tables))
                if tables:
                    st.write("**Table Names:**", ", ".join(tables))

def debug_table_info():
    """Debug function to show table information"""
    if st.session_state.get('connected') and st.session_state.sam:
        with st.expander("🔍 Debug: Table Information", expanded=False):
            tables = st.session_state.sam.get_tables()
            st.write(f"**Total Tables:** {len(tables)}")
            
            for table in tables:
                try:
                    schema = st.session_state.sam._get_table_schema(table)
                    st.write(f"**{table}:** {len(schema.columns)} columns, {schema.row_count} rows")
                # Replaced 'except Exception as e:' with 'except Exception as _:' as 'e' was unused
                except Exception as _:
                    st.write(f"**{table}:** Error loading - {_}")

def debug_schema_info():
    """Debug function to show schema file information"""
    if st.session_state.get('connected') and st.session_state.sam:
        with st.expander("🔍 Debug: Schema File", expanded=False):
            schema_file = st.session_state.sam.schema_file
            st.write("**Schema File Path:**", schema_file)
            st.write("**File Exists:**", os.path.exists(schema_file))
            
            if os.path.exists(schema_file):
                try:
                    with open(schema_file, 'r') as f:
                        content = f.read()
                    st.write("**File Size:**", len(content), "characters")
                    st.text_area("**Schema Content (first 1000 chars):**", content[:1000], height=200)
                # Replaced 'except Exception as e:' with 'except Exception as _:' as 'e' was unused
                except Exception as _:
                    st.error(f"Error reading schema file: {_}")

def display_debug_console():
    """Display real-time debug console for monitoring all feedback"""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("### 🖥️ Debug Console")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_debug_logs"):
            pass  # Refresh on button click
    
    with col3:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_debug_logs"):
            st.session_state.debug_logger.clear()
            st.rerun()
    
    # Get logs from debug logger
    logger = st.session_state.debug_logger
    logs = logger.get_logs()
    
    if logs:
        # Display logs in a code block with colors
        log_display = ""
        for log in logs:
            timestamp = log['timestamp']
            level = log['level']
            message = log['message']
            
            # Color coding based on level
            if level == "ERROR":
                emoji = "❌"
                color_indicator = "🔴"
            elif level == "SUCCESS":
                emoji = "✅"
                color_indicator = "🟢"
            elif level == "WARNING":
                emoji = "⚠️"
                color_indicator = "🟡"
            else:
                emoji = "ℹ️"
                color_indicator = "🔵"
            
            log_display += f"{color_indicator} [{timestamp}] {level}: {message}\n"
        
        # Display in a scrollable code block
        st.code(log_display, language="text")
        
        # Show log summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            errors = len([l for l in logs if l['level'] == 'ERROR'])
            st.metric("Errors", errors)
        with col2:
            successes = len([l for l in logs if l['level'] == 'SUCCESS'])
            st.metric("Success", successes)
        with col3:
            warnings = len([l for l in logs if l['level'] == 'WARNING'])
            st.metric("Warnings", warnings)
        with col4:
            infos = len([l for l in logs if l['level'] == 'INFO'])
            st.metric("Info", infos)
    else:
        st.info("No logs yet. Start an operation to see feedback here!")

def display_analysis_mode():
    """Display the advanced data analysis interface"""
    st.header("🔬 AI Data Analysis Mode")
    
    # Initialize analyzer
    if not st.session_state.data_analyzer and st.session_state.sam:
        schema = st.session_state.sam._get_database_name() or "Unknown"
        st.session_state.data_analyzer = DataAnalyzer(
            schema_snapshot=st.session_state.sam.get_tables().__str__(),
            api_key=api_key
        )
    
    # Table selection
    st.subheader("📊 Step 1: Select Tables for Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 All Tables", use_container_width=True, key="analysis_all_tables"):
            if st.session_state.sam:
                st.session_state.selected_analysis_tables = st.session_state.sam.get_tables()
                st.success("All tables selected")
    
    with col2:
        if st.button("❌ No Tables", use_container_width=True, key="analysis_no_tables"):
            st.session_state.selected_analysis_tables = []
            st.info("No tables selected")
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True, key="analysis_refresh"):
            st.rerun()
    
    # Custom table selection
    if st.session_state.sam:
        available_tables = st.session_state.sam.get_tables()
        st.session_state.selected_analysis_tables = st.multiselect(
            "Select Specific Tables:",
            available_tables,
            default=st.session_state.selected_analysis_tables
        )
    
    if st.session_state.selected_analysis_tables:
        st.success(f"✅ Selected {len(st.session_state.selected_analysis_tables)} table(s)")
        st.write(st.session_state.selected_analysis_tables)
    else:
        st.warning("⚠️ Please select at least one table")
    
    # Analysis query
    st.markdown("---")
    st.subheader("🤔 Step 2: Ask Your Question")
    
    analysis_query = st.text_area(
        "What would you like to know about your data?",
        value=st.session_state.current_analysis_query,
        placeholder="e.g., What is the distribution of student ages? Find the top performing students. Show me trends over time.",
        height=80
    )
    
    col_analyze1, col_analyze2 = st.columns([3, 1])
    with col_analyze1:
        analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True, key="analysis_start")
    
    with col_analyze2:
        if st.button("📜 History", use_container_width=True, key="analysis_history_btn"):
            st.session_state['show_analysis_history'] = True
    
    # Run analysis
    if analyze_btn and analysis_query and st.session_state.selected_analysis_tables:
        st.session_state.current_analysis_query = analysis_query
        
        logger = st.session_state.get('debug_logger')
        if logger:
            logger.info(f"Analysis query: {analysis_query}")
            logger.info(f"Selected tables: {', '.join(st.session_state.selected_analysis_tables)}")
        
        with st.spinner("🔬 AI is analyzing your data..."):
            try:
                if st.session_state.data_analyzer and st.session_state.executor:
                    analysis_session = st.session_state.data_analyzer.analyze(
                        user_query=analysis_query,
                        selected_tables=st.session_state.selected_analysis_tables,
                        executor=st.session_state.executor
                    )
                    
                    st.session_state.current_analysis_session = analysis_session
                    st.session_state.analysis_history.append(analysis_session.session_id)
                    
                    if logger:
                        logger.success(f"Analysis completed: {len(analysis_session.tasks)} tasks executed")
                    
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger = st.session_state.get('debug_logger')
                if logger:
                    logger.error(f"Analysis failed: {str(e)}")
    
    # Display analysis results
    if st.session_state.current_analysis_session:
        display_analysis_results(st.session_state.current_analysis_session)

def display_analysis_results(session: AnalysisSession):
    """Display analysis results step-by-step"""
    st.markdown("---")
    st.subheader("📈 Step 3: Analysis Results")
    
    # Final answer
    st.info(f"### 💡 Answer\n{session.final_answer}")
    
    # Tasks breakdown
    st.subheader("📋 Analysis Steps")
    
    for idx, task in enumerate(session.tasks, 1):
        with st.expander(f"Step {idx}: {task.type.value.upper()} - {task.description}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_emoji = {
                    "completed": "✅",
                    "failed": "❌",
                    "in_progress": "⏳",
                    "pending": "⏱️",
                    "refined": "🔧"
                }.get(task.status.value, "❓")
                st.metric("Status", f"{status_emoji} {task.status.value}")
            
            with col2:
                st.metric("Execution Time", f"{task.execution_time_ms:.0f}ms")
            
            with col3:
                st.metric("Tokens Used", task.tokens_used)
            
            # SQL Query
            if task.sql_query:
                st.code(task.sql_query, language="sql")
            
            # Results
            if task.result:
                st.subheader("Results")
                if isinstance(task.result, list):
                    df_result = pd.DataFrame(task.result)
                    st.dataframe(df_result, use_container_width=True)
                else:
                    st.json(task.result)
            
            # Refinement
            if task.status == TaskStatus.COMPLETED:
                if st.button(f"🔧 Refine Step {idx}", key=f"refine_{idx}"):
                    refinement = st.text_input(f"How should we refine this step?", key=f"refine_input_{idx}")
                    if refinement:
                        task.refinement_prompt = refinement
                        task.status = TaskStatus.REFINED
                        st.success("Refinement noted!")
            
            # Error handling
            if task.error:
                st.error(f"Error: {task.error}")
    
    # Summary metrics
    st.markdown("---")
    st.subheader("📊 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tasks", len(session.tasks))
    
    with col2:
        completed = len([t for t in session.tasks if t.status == TaskStatus.COMPLETED])
        st.metric("Completed", completed)
    
    with col3:
        failed = len([t for t in session.tasks if t.status == TaskStatus.FAILED])
        st.metric("Failed", failed)
    
    with col4:
        st.metric("Total Tokens", session.total_tokens_used)
    
    # Export results
    if st.button("📥 Export Analysis Results", key="export_analysis_results"):
        export_data = json.dumps(session.to_dict(), indent=2)
        st.download_button(
            label="Download JSON",
            data=export_data,
            file_name=f"analysis_{session.session_id}.json",
            mime="application/json"
        )

def main():

    """Enhanced main application with optional debug features"""
    try:
        initialize_session_state()
        display_enhanced_header()
        
        sidebar_database_connection()
        
        # Mode Toggle
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Mode Selection")
        mode_col1, mode_col2 = st.sidebar.columns(2)
        
        with mode_col1:
            if st.button("🔍 Query Mode", use_container_width=True, key="mode_query",
                        type="primary" if not st.session_state.analysis_mode else "secondary"):
                st.session_state.analysis_mode = False
                st.rerun()
        
        with mode_col2:
            if st.button("🔬 Analysis Mode", use_container_width=True, key="mode_analysis",
                        type="primary" if st.session_state.analysis_mode else "secondary"):
                st.session_state.analysis_mode = True
                st.rerun()
        
        # Debug Console Toggle
        st.sidebar.markdown("---")
        if st.sidebar.button("🖥️ Toggle Debug Console", use_container_width=True, key="toggle_debug_console"):
            st.session_state.show_debug_console = not st.session_state.show_debug_console
        
        # Display debug console if enabled
        if st.session_state.show_debug_console:
            st.markdown("---")
            display_debug_console()
            st.markdown("---")

        #Admin Console Access
        st.sidebar.markdown("---")
        st.sidebar.write(" 🔐 Admin Console")
        if st.sidebar.button("Open Admin Console", key="open_admin_console"):
            st.switch_page("pages/admin_console.py")

        # Debug sections (remove in production)
        if st.session_state.get('connected'):
            debug_schema_info()
            debug_database_status()
            debug_table_info()

        if st.session_state.connected:
            # Choose mode
            if st.session_state.analysis_mode:
                # Analysis Mode
                display_analysis_mode()
            else:
                # Query Mode (original)
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🔍 Query", 
                    "📋 Schema", 
                    "📜 History", 
                    "📈 Analytics"
                ])
                
                with tab1:
                    display_query_interface()
                
                with tab2:
                    display_schema_explorer()
                
                with tab3:
                    display_query_history()
                
                with tab4:
                    display_analytics_dashboard()
        
        else:
            display_welcome_screen()
            
    # Replaced 'except Exception as e:' with 'except Exception as _:' as 'e' was unused
    except Exception as _:
        st.error(f"Application error: {_}")
        import traceback
        st.code(traceback.format_exc())
              
if __name__ == "__main__":
    main()