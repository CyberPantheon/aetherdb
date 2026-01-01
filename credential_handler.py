#!/usr/bin/env python3
"""
Credential Handler Module

Provides robust handling of database credentials with support for edge cases:
- Databases without passwords
- Empty/blank passwords
- Optional authentication
- Secure credential validation and sanitization

This module ensures the system can accommodate various database authentication 
scenarios without requiring all fields to be filled.
"""

from typing import Dict, Optional, Any
import re


class CredentialHandler:
    """
    Manages database credentials robustly, handling various authentication scenarios.
    """
    
    @staticmethod
    def sanitize_credentials(credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize and normalize database credentials.
        
        Args:
            credentials: Dictionary containing connection parameters
            
        Returns:
            Sanitized credentials dictionary
        """
        sanitized = {}
        
        # Required fields
        if 'host' in credentials:
            sanitized['host'] = str(credentials['host']).strip() or 'localhost'
        if 'user' in credentials:
            sanitized['user'] = str(credentials['user']).strip()
        if 'database' in credentials:
            sanitized['database'] = str(credentials['database']).strip()
        if 'port' in credentials:
            try:
                sanitized['port'] = int(credentials['port'])
            except (ValueError, TypeError):
                sanitized['port'] = None
        
        # Optional password - only include if provided and non-empty
        if 'password' in credentials:
            password = credentials['password']
            if isinstance(password, str) and password.strip():
                sanitized['password'] = password
        
        return sanitized
    
    @staticmethod
    def validate_connection_params(
        db_type: str,
        params: Dict[str, Any],
        allow_no_password: bool = True
    ) -> tuple[bool, Optional[str]]:
        """
        Validate database connection parameters.
        
        Args:
            db_type: Database type (mysql, postgres, sqlite)
            params: Connection parameters
            allow_no_password: Whether to allow connections without password
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        db_type_lower = db_type.lower()
        
        if db_type_lower == 'sqlite':
            # SQLite only requires database path
            if 'database' not in params or not params['database']:
                return False, "Database file path is required for SQLite"
            return True, None
        
        # For MySQL and PostgreSQL
        if db_type_lower not in ['mysql', 'postgres', 'postgresql']:
            return False, f"Unsupported database type: {db_type}"
        
        # Check required fields
        if 'user' not in params or not params['user']:
            return False, "Username is required"
        
        if 'database' not in params or not params['database']:
            return False, "Database name is required"
        
        # Validate host
        if 'host' in params and params['host']:
            host = str(params['host']).strip()
            if not CredentialHandler._is_valid_host(host):
                return False, f"Invalid host: {host}"
        
        # Validate port if provided
        if 'port' in params and params['port']:
            try:
                port = int(params['port'])
                if port < 1 or port > 65535:
                    return False, f"Port must be between 1 and 65535, got {port}"
            except (ValueError, TypeError):
                return False, "Port must be a valid integer"
        
        # Password handling
        password = params.get('password', '').strip() if params.get('password') else ''
        
        if not password and not allow_no_password:
            return False, "Password is required for this database"
        
        return True, None
    
    @staticmethod
    def _is_valid_host(host: str) -> bool:
        """
        Validate hostname or IP address.
        
        Args:
            host: Hostname or IP address to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not host:
            return False
        
        # Allow localhost variations
        if host.lower() in ['localhost', '127.0.0.1', '::1']:
            return True
        
        # Basic validation for domain names and IP addresses
        # IPv4
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ipv4_pattern, host):
            parts = host.split('.')
            return all(0 <= int(p) <= 255 for p in parts)
        
        # IPv6
        if ':' in host:
            return True  # Basic IPv6 check
        
        # Domain name
        domain_pattern = r'^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$'
        return re.match(domain_pattern, host) is not None
    
    @staticmethod
    def get_connection_string(db_type: str, params: Dict[str, Any]) -> str:
        """
        Generate a display-safe connection string (without password).
        
        Args:
            db_type: Database type
            params: Connection parameters
            
        Returns:
            Connection string for logging/display
        """
        db_type_lower = db_type.lower()
        
        if db_type_lower == 'sqlite':
            return f"SQLite: {params.get('database', 'unknown')}"
        
        user = params.get('user', 'unknown')
        host = params.get('host', 'localhost')
        database = params.get('database', 'unknown')
        port = params.get('port', 3306 if db_type_lower == 'mysql' else 5432)
        
        return f"{db_type}://{user}@{host}:{port}/{database}"
    
    @staticmethod
    def mask_credentials(credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a copy of credentials with password masked for logging.
        
        Args:
            credentials: Credentials dictionary
            
        Returns:
            Copy with masked password
        """
        masked = credentials.copy()
        if 'password' in masked and masked['password']:
            masked['password'] = '***'
        return masked


class EdgeCaseHandler:
    """
    Handles specific database edge cases and compatibility issues.
    """
    
    @staticmethod
    def prepare_mysql_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare MySQL connection parameters, handling edge cases.
        
        Args:
            params: Raw connection parameters
            
        Returns:
            MySQL-compatible parameters
        """
        prepared = {
            'host': params.get('host', 'localhost'),
            'user': params.get('user'),
            'database': params.get('database'),
            'port': int(params.get('port', 3306)),
        }
        
        # Add password only if provided and non-empty
        password = params.get('password', '').strip()
        if password:
            prepared['password'] = password
        # Note: PyMySQL will use empty password if not provided, allowing no-auth connections
        
        return prepared
    
    @staticmethod
    def prepare_postgres_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare PostgreSQL connection parameters, handling edge cases.
        
        Args:
            params: Raw connection parameters
            
        Returns:
            PostgreSQL-compatible parameters
        """
        prepared = {
            'host': params.get('host', 'localhost'),
            'user': params.get('user'),
            'database': params.get('database'),
            'port': int(params.get('port', 5432)),
        }
        
        # Add password only if provided and non-empty
        password = params.get('password', '').strip()
        if password:
            prepared['password'] = password
        # Note: psycopg2 will use trust authentication if password not provided
        
        return prepared
    
    @staticmethod
    def prepare_sqlite_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare SQLite connection parameters.
        
        Args:
            params: Raw connection parameters
            
        Returns:
            SQLite-compatible parameters (minimal)
        """
        return {'database': params.get('database')}
    
    @staticmethod
    def handle_connection_error(
        error: Exception,
        db_type: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Generate user-friendly error message based on connection error type.
        
        Args:
            error: Exception that occurred
            db_type: Database type
            params: Connection parameters (will be masked)
            
        Returns:
            User-friendly error message
        """
        error_str = str(error)
        db_type_lower = db_type.lower()
        
        # Mask sensitive info
        if 'password' in params and params['password']:
            error_str = error_str.replace(params['password'], '***')
        
        # Common error patterns
        if 'access denied' in error_str.lower() or 'authentication failed' in error_str.lower():
            return (
                f"Authentication failed for user '{params.get('user')}'. "
                "Check username and password (or leave password blank if not required)."
            )
        
        if 'unknown database' in error_str.lower() or 'does not exist' in error_str.lower():
            return f"Database '{params.get('database')}' does not exist on the server."
        
        if 'connection refused' in error_str.lower():
            host = params.get('host', 'localhost')
            port = params.get('port', 3306 if db_type_lower == 'mysql' else 5432)
            return f"Connection refused. Check if {db_type} server is running at {host}:{port}"
        
        if 'unknown host' in error_str.lower():
            return f"Unknown host: {params.get('host', 'localhost')}"
        
        # Default message
        return f"Failed to connect to {db_type}: {error_str}"
