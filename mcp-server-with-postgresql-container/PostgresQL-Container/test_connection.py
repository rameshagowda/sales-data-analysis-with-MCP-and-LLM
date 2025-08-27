"""Test database connection."""
import psycopg2
from psycopg2.extras import RealDictCursor
from config import DB_CONFIG

def test_database_connection():
    """Test the database connection."""
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Create cursor
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Test query
        cursor.execute('SELECT version() as version;')
        result = cursor.fetchone()
        
        print("✅ Database connection successful!")
        print(f"PostgreSQL version: {result['version']}")
        
        # Test database and user
        cursor.execute('SELECT current_database(), current_user;')
        db_info = cursor.fetchone()
        print(f"Database: {db_info['current_database']}")
        print(f"User: {db_info['current_user']}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    test_database_connection()