"""Database configuration for Sales Analysis MCP Server."""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG: Dict[str, Any] = {
    'host': os.getenv('DATABASE_HOST', 'localhost'),
    'port': int(os.getenv('DATABASE_PORT', 5432)),
    'database': os.getenv('DATABASE_NAME', 'sales_production'),
    'user': os.getenv('DATABASE_USER', 'sales_user'),
    'password': os.getenv('DATABASE_PASSWORD', 'secure_password')
}

# Alternative: Use DATABASE_URL for connection
DATABASE_URL = os.getenv(
    'DATABASE_URL', 
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# Application settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')