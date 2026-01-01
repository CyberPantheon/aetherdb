#!/usr/bin/env python3
"""
Data Analyzer Module (DAM)

RAG-based context-aware AI data analyst that:
- Creates and executes analysis tasks sequentially/in parallel
- Maintains JSON history for task tracking
- Manages tokens and context efficiently
- Integrates with existing modules (sqlm, db_executor, schema_awareness)
- Provides step-by-step analysis walkthrough

Usage:
    analyzer = DataAnalyzer(schema_snapshot, api_key)
    analysis = analyzer.analyze(user_query, tables)
"""

from __future__ import annotations
import os
import json
import re
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
import sqlite3
from pathlib import Path
import contextlib
from decimal import Decimal

from dotenv import load_dotenv

load_dotenv()


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Decimal, date, and other non-serializable types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        return super().default(obj)


def _serialize_value(value: Any) -> Any:
    """Recursively convert non-serializable values to JSON-safe types"""
    if isinstance(value, Decimal):
        return float(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    elif hasattr(value, '__dict__'):
        return _serialize_value(value.__dict__)
    else:
        return value

# Import optional dependencies with fallbacks
try:
    import importlib
    _genai_mod = importlib.import_module("google.generativeai")
    genai = _genai_mod
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# Config from env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
MAX_ANALYSIS_CONTEXT = int(os.getenv("MAX_ANALYSIS_CONTEXT", "20000"))


class TaskStatus(Enum):
    """Status of an analysis task"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REFINED = "refined"


class TaskType(Enum):
    """Types of analysis tasks"""
    EXPLORATION = "exploration"  # Initial data exploration
    AGGREGATION = "aggregation"  # Group by, counts, sums
    FILTERING = "filtering"      # Find specific records
    CORRELATION = "correlation"  # Find relationships
    TREND = "trend"             # Time-based trends
    COMPARISON = "comparison"   # Compare groups
    OUTLIER = "outlier"         # Find anomalies
    SYNTHESIS = "synthesis"     # Combine findings


@dataclass
class AnalysisTask:
    """A single analysis task to execute"""
    id: str
    type: TaskType
    description: str
    sql_query: Optional[str] = None
    result: Optional[Any] = None
    status: TaskStatus = TaskStatus.PENDING
    error: Optional[str] = None
    tokens_used: int = 0
    execution_time_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    refinement_prompt: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['type'] = self.type.value
        result['status'] = self.status.value
        # Recursively convert non-serializable types
        result['result'] = _serialize_value(result.get('result'))
        return result


@dataclass
class AnalysisSession:
    """Complete analysis session with history"""
    session_id: str
    user_query: str
    selected_tables: List[str]
    tasks: List[AnalysisTask] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_tokens_used: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    schema_snapshot: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'session_id': self.session_id,
            'user_query': self.user_query,
            'selected_tables': self.selected_tables,
            'tasks': [task.to_dict() for task in self.tasks],
            'final_answer': self.final_answer,
            'total_tokens_used': self.total_tokens_used,
            'created_at': self.created_at,
            'completed_at': self.completed_at
        }


class DataAnalyzer:
    """
    RAG-based data analyzer for intelligent SQL generation and analysis
    """
    
    def __init__(
        self,
        schema_snapshot: str = "",
        api_key: Optional[str] = GEMINI_API_KEY,
        history_dir: str = "./analysis_history"
    ):
        """
        Initialize the Data Analyzer
        
        Args:
            schema_snapshot: Database schema for context
            api_key: Gemini API key
            history_dir: Directory to store analysis history
        """
        self.schema_snapshot = schema_snapshot or "NO_SCHEMA_PROVIDED"
        self.api_key = api_key
        self.model_name = GEMINI_MODEL
        self.model = None
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        
        self._init_client()
        print("[DAM] Data Analyzer initialized")
    
    def _init_client(self):
        """Initialize Gemini API client"""
        if self._use_real_genai():
            genai.configure(api_key=self.api_key)
            try:
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                print(f"Warning: Failed to init model {self.model_name}: {e}")
        else:
            self.model = None
    
    def _use_real_genai(self) -> bool:
        """Check if we can use real Gemini API"""
        return GENAI_AVAILABLE and bool(self.api_key)
    
    def update_schema(self, schema_snapshot: str):
        """Update the schema snapshot"""
        self.schema_snapshot = schema_snapshot
    
    def analyze(
        self,
        user_query: str,
        selected_tables: List[str],
        executor = None,
        refinement_callback: Optional[callable] = None
    ) -> AnalysisSession:
        """
        Perform comprehensive data analysis
        
        Args:
            user_query: Natural language question
            selected_tables: Tables to analyze
            executor: DatabaseExecutor instance for running queries
            refinement_callback: Function to call for user refinement
            
        Returns:
            AnalysisSession with tasks and results
        """
        session = AnalysisSession(
            session_id=self._generate_session_id(),
            user_query=user_query,
            selected_tables=selected_tables,
            schema_snapshot=self.schema_snapshot
        )
        
        print(f"[DAM] Starting analysis session: {session.session_id}")
        
        try:
            # Step 1: Generate analysis tasks
            tasks = self._generate_tasks(user_query, selected_tables)
            session.tasks = tasks
            
            # Step 2: Execute tasks sequentially with context awareness
            for task in session.tasks:
                if task.status == TaskStatus.PENDING:
                    self._execute_task(task, executor, session)
                    session.total_tokens_used += task.tokens_used
            
            # Step 3: Generate final synthesis answer
            session.final_answer = self._synthesize_answer(session)
            session.completed_at = datetime.now().isoformat()
            
            # Step 4: Save session history
            self._save_session(session)
            
            print(f"[DAM] Analysis completed. Tokens used: {session.total_tokens_used}")
            
            return session
            
        except Exception as e:
            print(f"[DAM] Analysis failed: {e}")
            session.final_answer = f"Analysis failed: {str(e)}"
            return session
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        hash_obj = hashlib.md5(timestamp.encode())
        return f"session_{hash_obj.hexdigest()[:8]}"
    
    def _generate_tasks(
        self,
        user_query: str,
        selected_tables: List[str]
    ) -> List[AnalysisTask]:
        """
        Generate analysis tasks based on user query
        Uses RAG to understand intent and create appropriate tasks
        
        Args:
            user_query: Natural language question
            selected_tables: Available tables
            
        Returns:
            List of tasks to execute
        """
        prompt = self._build_task_generation_prompt(user_query, selected_tables)
        
        try:
            response = self._call_llm(prompt)
            tasks = self._parse_task_response(response, user_query)
            return tasks
        except Exception as e:
            print(f"[DAM] Task generation failed: {e}")
            # Fallback: create a basic exploration task
            return [
                AnalysisTask(
                    id="task_0",
                    type=TaskType.EXPLORATION,
                    description=f"Explore data to answer: {user_query}"
                )
            ]
    
    def _build_task_generation_prompt(self, query: str, tables: List[str]) -> str:
        """Build prompt for task generation"""
        tables_str = ", ".join(tables) if tables else "all available tables"
        
        return f"""You are an expert data analyst. Analyze the user's question and create a step-by-step analysis plan.

USER QUESTION: {query}
AVAILABLE TABLES: {tables_str}

SCHEMA:
{self.schema_snapshot[:5000]}

Create a JSON array of analysis tasks. Each task should:
1. Have a type: exploration, aggregation, filtering, correlation, trend, comparison, outlier, or synthesis
2. Include a clear description of what to do
3. Include the SQL query needed (or null if pending user refinement)

Example format:
[
  {{
    "type": "exploration",
    "description": "First, get basic statistics about the dataset",
    "sql_query": "SELECT COUNT(*) as total_records FROM students"
  }},
  {{
    "type": "aggregation",
    "description": "Group data and find patterns",
    "sql_query": "SELECT age, COUNT(*) FROM students GROUP BY age"
  }},
  {{
    "type": "synthesis",
    "description": "Combine findings to answer the question",
    "sql_query": null
  }}
]

Return ONLY the JSON array:"""
    
    def _parse_task_response(self, response: str, query: str) -> List[AnalysisTask]:
        """Parse LLM response into AnalysisTask objects"""
        try:
            # Extract JSON
            text = response.strip()
            s, e = text.find("["), text.rfind("]")
            if s != -1 and e != -1:
                json_text = text[s:e+1]
                tasks_data = json.loads(json_text)
            else:
                return []
            
            tasks = []
            for idx, task_data in enumerate(tasks_data):
                task = AnalysisTask(
                    id=f"task_{idx}",
                    type=TaskType[task_data.get('type', 'EXPLORATION').upper()],
                    description=task_data.get('description', ''),
                    sql_query=task_data.get('sql_query')
                )
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            print(f"[DAM] Task parsing failed: {e}")
            return []
    
    def _execute_task(
        self,
        task: AnalysisTask,
        executor,
        session: AnalysisSession
    ):
        """
        Execute a single analysis task
        
        Args:
            task: AnalysisTask to execute
            executor: DatabaseExecutor instance
            session: Current AnalysisSession for context
        """
        import time
        start_time = time.time()
        task.status = TaskStatus.IN_PROGRESS
        
        try:
            # If SQL query is provided, execute it
            if task.sql_query and executor:
                result = executor.execute_query(task.sql_query, safe_to_execute=True)
                task.result = result.data if hasattr(result, 'data') else result
            else:
                # Generate SQL using context
                task.sql_query = self._generate_task_sql(task, session)
                
                if task.sql_query and executor:
                    result = executor.execute_query(task.sql_query, safe_to_execute=True)
                    task.result = result.data if hasattr(result, 'data') else result
            
            task.status = TaskStatus.COMPLETED
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            print(f"[DAM] Task {task.id} failed: {e}")
        
        finally:
            task.execution_time_ms = (time.time() - start_time) * 1000
            task.completed_at = datetime.now().isoformat()
    
    def _generate_task_sql(self, task: AnalysisTask, session: AnalysisSession) -> Optional[str]:
        """
        Generate SQL for a task using AI
        
        Args:
            task: Task to generate SQL for
            session: Current analysis session
            
        Returns:
            Generated SQL query or None
        """
        prompt = f"""Based on this analysis task, generate appropriate SQL query:

TASK: {task.description}
TASK TYPE: {task.type.value}

SCHEMA:
{self.schema_snapshot[:3000]}

TABLES TO USE: {', '.join(session.selected_tables)}

Generate ONLY the SQL query (no explanation):"""
        
        try:
            response = self._call_llm(prompt)
            # Extract SQL from response
            sql = response.strip()
            # Remove markdown if present
            if sql.startswith("```"):
                sql = sql.split("```")[1].strip()
                if sql.startswith("sql"):
                    sql = sql[3:].strip()
            return sql
        except Exception as e:
            print(f"[DAM] SQL generation for task failed: {e}")
            return None
    
    def _synthesize_answer(self, session: AnalysisSession) -> str:
        """
        Synthesize final answer from task results
        
        Args:
            session: Analysis session with completed tasks
            
        Returns:
            Natural language answer
        """
        # Collect all results
        results_summary = self._summarize_results(session.tasks)
        
        prompt = f"""Based on the analysis results below, provide a clear, comprehensive answer to the user's question.

USER QUESTION: {session.user_query}

ANALYSIS RESULTS:
{results_summary}

Provide a natural language summary that:
1. Directly answers the question
2. Highlights key findings
3. Mentions any patterns or anomalies
4. Is concise and clear"""
        
        try:
            answer = self._call_llm(prompt)
            return answer.strip()
        except Exception as e:
            print(f"[DAM] Synthesis failed: {e}")
            return "Analysis completed. Check individual task results."
    
    def _summarize_results(self, tasks: List[AnalysisTask]) -> str:
        """Summarize task results for synthesis"""
        summary = []
        for task in tasks:
            if task.status == TaskStatus.COMPLETED and task.result:
                summary.append(f"- {task.description}: {self._format_result(task.result)}")
            elif task.status == TaskStatus.FAILED:
                summary.append(f"- {task.description}: Failed - {task.error}")
        
        return "\n".join(summary) if summary else "No results available"
    
    def _format_result(self, result: Any) -> str:
        """Format result for display"""
        if isinstance(result, list):
            if len(result) > 5:
                return f"{len(result)} records found. Sample: {result[:2]}"
            return str(result)
        return str(result)
    
    def _call_llm(self, prompt: str) -> str:
        """Call Gemini API or fallback"""
        if not self._use_real_genai() or self.model is None:
            return "Mock response"
        
        try:
            response = self.model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                return response.text
            elif hasattr(response, "candidates") and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                raise RuntimeError("No text in response")
        except Exception as e:
            raise RuntimeError(f"API call failed: {e}")
    
    def _save_session(self, session: AnalysisSession):
        """Save session history to JSON file"""
        try:
            session_file = self.history_dir / f"{session.session_id}.json"
            session_data = session.to_dict()
            # Serialize all values recursively to handle Decimal and other types
            session_data = _serialize_value(session_data)
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, cls=JSONEncoder)
            print(f"[DAM] Session saved to {session_file}")
        except Exception as e:
            print(f"[DAM] Failed to save session: {e}")
    
    def load_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Load a previous session from history"""
        try:
            session_file = self.history_dir / f"{session_id}.json"
            if session_file.exists():
                with open(session_file, 'r') as f:
                    data = json.load(f)
                # Reconstruct AnalysisSession from dict
                session = AnalysisSession(
                    session_id=data['session_id'],
                    user_query=data['user_query'],
                    selected_tables=data['selected_tables'],
                    final_answer=data.get('final_answer'),
                    total_tokens_used=data.get('total_tokens_used', 0),
                    created_at=data['created_at'],
                    completed_at=data.get('completed_at')
                )
                return session
        except Exception as e:
            print(f"[DAM] Failed to load session: {e}")
        
        return None
    
    def get_history(self, limit: int = 10) -> List[str]:
        """Get list of recent session IDs"""
        try:
            sessions = sorted(
                [f.stem for f in self.history_dir.glob("*.json")],
                reverse=True
            )
            return sessions[:limit]
        except Exception:
            return []


if __name__ == "__main__":
    # Example usage
    analyzer = DataAnalyzer()
    
    # Mock session
    session = AnalysisSession(
        session_id="test_001",
        user_query="What is the average age of students?",
        selected_tables=["students"]
    )
    
    print(json.dumps(session.to_dict(), indent=2))
