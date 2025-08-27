#!/usr/bin/env python3
"""
Complete Sales Analysis Database Tool (PostgreSQL Version)
"""

import psycopg2
import psycopg2.extras
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sales_production")

# PostgreSQL connection parameters (hardcoded for simplicity)
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'sales_production',
    'user': 'sales_user',
    'password': 'secure_password'
}

# Table names constants
CUSTOMERS_TABLE = "customers"
PRODUCTS_TABLE = "products"
ORDERS_TABLE = "orders"
ORDER_ITEMS_TABLE = "order_items"
STORES_TABLE = "stores"
CATEGORIES_TABLE = "categories"
PRODUCT_TYPES_TABLE = "product_types"
INVENTORY_TABLE = "inventory"


class PostgreSQLSchemaProvider:
    """Provides PostgreSQL database schema information in AI-friendly formats."""

    def __init__(self, db_config: Optional[Dict] = None) -> None:
        self.db_config = db_config or DB_CONFIG
        self.all_schemas: Optional[Dict[str, Dict[str, Any]]] = None
        self._schema_cache: Dict[str, Any] = {}

    def get_connection(self) -> psycopg2.extensions.connection:
        """Return a new connection to the PostgreSQL database."""
        conn = psycopg2.connect(**self.db_config)
        conn.set_session(autocommit=False)
        return conn

    @staticmethod
    def _parse_table_name(table: str) -> Tuple[str, str]:
        """Accept either 'schema.table' or 'table'. Return (schema, table)."""
        if "." in table:
            parts = table.split(".", 1)
            return parts[0], parts[1]
        return "public", table

    def table_exists(self, table: str) -> bool:
        """Check if a table exists in PostgreSQL."""
        schema_name, table_name = self._parse_table_name(table)
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s AND table_name = %s
                )
            """, (schema_name, table_name))
            exists = cursor.fetchone()[0]
            conn.close()
            return exists
        except Exception:
            return False

    def column_exists(self, table: str, column: str) -> bool:
        """Check if a column exists in a PostgreSQL table."""
        schema_name, table_name = self._parse_table_name(table)
        if not self.table_exists(table):
            return False
        
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_schema = %s AND table_name = %s AND column_name = %s
            )
        """, (schema_name, table_name, column))
        exists = cursor.fetchone()[0]
        conn.close()
        return exists

    def fetch_distinct_values(self, column: str, table: str) -> List[str]:
        """Return sorted distinct values for a column in a table."""
        schema_name, table_name = self._parse_table_name(table)
        if not self.table_exists(table):
            raise ValueError(f"Table '{table}' does not exist")
        if not self.column_exists(table, column):
            raise ValueError(f"Column '{column}' does not exist in table '{table_name}'")

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f'SELECT DISTINCT "{column}" FROM "{schema_name}"."{table_name}" WHERE "{column}" IS NOT NULL ORDER BY "{column}" ASC'
        )
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows if row[0] is not None]

    def infer_relationship_type(self, references_table: str) -> str:
        """Infer relationship type based on table name."""
        try:
            _, table_name = self._parse_table_name(references_table)
        except Exception:
            table_name = references_table

        return (
            "many_to_one"
            if table_name
            in {CUSTOMERS_TABLE, PRODUCTS_TABLE, STORES_TABLE, CATEGORIES_TABLE, PRODUCT_TYPES_TABLE, ORDERS_TABLE}
            else "one_to_many"
        )

    def get_table_schema(self, table_name: str, rls_user_id: str = "") -> Dict[str, Any]:
        """Return schema information for a given table."""
        if table_name in self._schema_cache:
            return self._schema_cache[table_name]

        schema_part, parsed_table_name = self._parse_table_name(table_name)

        if not self.table_exists(table_name):
            return {"error": f"Table '{table_name}' not found"}

        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Get column information
        cursor.execute("""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length
            FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """, (schema_part, parsed_table_name))
        columns_info = cursor.fetchall()

        # Get primary keys
        cursor.execute("""
            SELECT column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = %s
                AND tc.table_name = %s
        """, (schema_part, parsed_table_name))
        pk_rows = cursor.fetchall()
        pk_columns = {row['column_name'] for row in pk_rows}

        # Get foreign keys
        cursor.execute("""
            SELECT
                kcu.column_name,
                ccu.table_name AS references_table,
                ccu.column_name AS references_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = %s
                AND tc.table_name = %s
        """, (schema_part, parsed_table_name))
        fk_rows = cursor.fetchall()

        foreign_keys = [
            {
                "column": row["column_name"],
                "references_table": row["references_table"],
                "references_column": row["references_column"],
                "description": f"{row['column_name']} links to {row['references_table']}.{row['references_column']}",
                "relationship_type": self.infer_relationship_type(row["references_table"]),
            }
            for row in fk_rows
        ]

        # Build columns_format string
        columns_format = ", ".join(f"{col['column_name']}:{col['data_type']}" for col in columns_info)

        lower_table = parsed_table_name.lower()

        # Enum-ish queries mapping
        enum_queries = {
            STORES_TABLE: {"available_stores": ("store_name", parsed_table_name)},
            CATEGORIES_TABLE: {"available_categories": ("category_name", parsed_table_name)},
            PRODUCT_TYPES_TABLE: {"available_product_types": ("type_name", parsed_table_name)},
            ORDERS_TABLE: {"available_years": ("EXTRACT(YEAR FROM order_date)", parsed_table_name)},
        }

        enum_data: Dict[str, Any] = {}
        if lower_table in enum_queries:
            for key, (column_expr, tbl) in enum_queries[lower_table].items():
                try:
                    if key == "available_years":
                        cursor.execute(f'SELECT DISTINCT {column_expr} as year FROM "{schema_part}"."{tbl}" WHERE order_date IS NOT NULL ORDER BY year')
                        rows = cursor.fetchall()
                        enum_data[key] = [str(int(r['year'])) for r in rows if r['year'] is not None]
                    else:
                        cursor.execute(f'SELECT DISTINCT {column_expr} FROM "{schema_part}"."{tbl}" WHERE {column_expr} IS NOT NULL ORDER BY {column_expr}')
                        rows = cursor.fetchall()
                        enum_data[key] = [r[column_expr.split('.')[-1]] for r in rows if r[column_expr.split('.')[-1]] is not None]
                except Exception as e:
                    logger.debug(f"Failed to fetch {key} for {tbl}: {e}")
                    enum_data[key] = []

        schema_data = {
            "table_name": table_name,
            "parsed_table_name": parsed_table_name,
            "schema_name": schema_part,
            "description": f"Table containing {parsed_table_name} data",
            "columns_format": columns_format,
            "columns": [
                {
                    "name": col["column_name"],
                    "type": col["data_type"],
                    "primary_key": col["column_name"] in pk_columns,
                    "required": col["is_nullable"] == "NO",
                    "default_value": col["column_default"],
                }
                for col in columns_info
            ],
            "foreign_keys": foreign_keys,
        }

        schema_data.update(enum_data)
        self._schema_cache[table_name] = schema_data
        conn.close()
        return schema_data

    def get_all_table_names(self) -> List[str]:
        """Return all table names in the public schema."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows]

    def get_all_schemas(self, schema_name: str = "", rls_user_id: str = "") -> Dict[str, Dict[str, Any]]:
        """Build a dictionary of all table schemas."""
        table_names = self.get_all_table_names()
        result: Dict[str, Dict[str, Any]] = {}
        for tbl in table_names:
            schema_data = self.get_table_schema(tbl, rls_user_id=rls_user_id)
            result[tbl] = schema_data
        self.all_schemas = result
        return result

    def format_schema_metadata_for_ai(self, schema: Dict[str, Any]) -> str:
        """Format schema dictionary into an AI-friendly human-readable string."""
        if "error" in schema:
            return f"**ERROR:** {schema['error']}\n"

        table_display = schema.get("table_name")
        try:
            _, table_name_only = self._parse_table_name(table_display) if table_display else ("", "unknown")
            table_description = table_name_only.replace("_", " ")
        except Exception:
            table_description = (table_display or "unknown").replace("_", " ")

        lines: List[str] = [f"# Table: {table_display}", ""]
        lines.append(f"**Purpose:** {schema.get('description', 'No description available')}")
        lines.append("\n## Schema")
        lines.append(schema.get("columns_format", "N/A"))

        if schema.get("foreign_keys"):
            lines.append("\n## Relationships")
            for fk in schema["foreign_keys"]:
                fk_table_ref = fk.get("references_table", "unknown")
                lines.append(f"- `{fk['column']}` → `{fk_table_ref}.{fk['references_column']}` ({fk['relationship_type'].upper()})")

        enum_fields = [
            ("available_stores", "Stores Locations"),
            ("available_categories", "Valid Categories"),
            ("available_product_types", "Valid Product Types"),
            ("available_years", "Available Years"),
        ]

        enum_lines: List[str] = []
        for field_key, label in enum_fields:
            if schema.get(field_key):
                values = schema[field_key]
                enum_lines.append(f"**{label}:** {', '.join(values) if isinstance(values, list) else str(values)}")

        if enum_lines:
            lines.append("\n## Valid Values")
            lines.extend(enum_lines)

        lines.append("\n## Query Hints")
        lines.append(f"- Use `{table_display}` for queries about {table_description}")
        if schema.get("foreign_keys"):
            for fk in schema["foreign_keys"]:
                fk_table_ref = fk.get("references_table", "unknown")
                lines.append(f"- Join with `{fk_table_ref}` using `{fk['column']}`")

        return "\n".join(lines) + "\n"

    def get_table_metadata_string(self, table_name: str, rls_user_id: str = "") -> str:
        """Return formatted schema metadata string for a single table."""
        schema = self.get_table_schema(table_name, rls_user_id=rls_user_id)
        return self.format_schema_metadata_for_ai(schema)

    def get_table_metadata_from_list(self, table_names: List[str], rls_user_id: str = "") -> str:
        """Return formatted schema metadata strings for multiple tables."""
        if not table_names:
            return "Error: table_names parameter is required and cannot be empty"

        schemas: List[str] = []
        for qualified_name in table_names:
            try:
                schema_name, parsed_table_name = self._parse_table_name(qualified_name)
                if not self.table_exists(qualified_name):
                    schemas.append(f"**ERROR:** Table '{qualified_name}' not found\n")
                    continue
                schema_data = self.get_table_schema(qualified_name, rls_user_id=rls_user_id)
                schemas.append(self.format_schema_metadata_for_ai(schema_data))
            except Exception as e:
                schemas.append(f"Error retrieving {qualified_name} schema: {e!s}\n")

        return "\n\n".join(schemas)

    def execute_query(self, sql_query: str, rls_user_id: str = "") -> str:
        """Execute SQL query against PostgreSQL and return LLM-friendly JSON string."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute(sql_query)
            rows = cursor.fetchall()

            if not rows:
                result = {
                    "results": [],
                    "row_count": 0,
                    "columns": [],
                    "message": "The query returned no results. Try a different question.",
                }
            else:
                results = [dict(row) for row in rows]
                columns = list(results[0].keys()) if results else []
                result = {
                    "results": results,
                    "row_count": len(results),
                    "columns": columns
                }

            conn.close()
            return json.dumps(result, indent=2, default=str)

        except Exception as e:
            return json.dumps(
                {
                    "error": f"PostgreSQL query failed: {e!s}",
                    "query": sql_query,
                    "results": [],
                    "row_count": 0,
                    "columns": [],
                }
            )


def init_db():
    """Initialize the PostgreSQL database with all tables and sample data"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Create tables with proper PostgreSQL data types
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {STORES_TABLE} (
                store_id SERIAL PRIMARY KEY,
                store_name VARCHAR(100) NOT NULL,
                location VARCHAR(200),
                manager VARCHAR(100)
            )
        """)
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {CATEGORIES_TABLE} (
                category_id SERIAL PRIMARY KEY,
                category_name VARCHAR(100) NOT NULL,
                description TEXT
            )
        """)
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {PRODUCT_TYPES_TABLE} (
                type_id SERIAL PRIMARY KEY,
                type_name VARCHAR(50) NOT NULL,
                description TEXT
            )
        """)
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {PRODUCTS_TABLE} (
                product_id SERIAL PRIMARY KEY,
                product_name VARCHAR(200) NOT NULL,
                category_id INTEGER REFERENCES {CATEGORIES_TABLE}(category_id),
                type_id INTEGER REFERENCES {PRODUCT_TYPES_TABLE}(type_id),
                price DECIMAL(10,2)
            )
        """)
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {CUSTOMERS_TABLE} (
                customer_id SERIAL PRIMARY KEY,
                first_name VARCHAR(50) NOT NULL,
                last_name VARCHAR(50) NOT NULL,
                email VARCHAR(100),
                phone VARCHAR(20)
            )
        """)
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {ORDERS_TABLE} (
                order_id SERIAL PRIMARY KEY,
                customer_id INTEGER REFERENCES {CUSTOMERS_TABLE}(customer_id),
                store_id INTEGER REFERENCES {STORES_TABLE}(store_id),
                order_date DATE,
                total_amount DECIMAL(10,2)
            )
        """)
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {ORDER_ITEMS_TABLE} (
                order_item_id SERIAL PRIMARY KEY,
                order_id INTEGER REFERENCES {ORDERS_TABLE}(order_id),
                product_id INTEGER REFERENCES {PRODUCTS_TABLE}(product_id),
                quantity INTEGER,
                unit_price DECIMAL(10,2),
                total_amount DECIMAL(10,2)
            )
        """)
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {INVENTORY_TABLE} (
                inventory_id SERIAL PRIMARY KEY,
                product_id INTEGER REFERENCES {PRODUCTS_TABLE}(product_id),
                store_id INTEGER REFERENCES {STORES_TABLE}(store_id),
                quantity_in_stock INTEGER,
                reorder_level INTEGER
            )
        """)
        
        # Check if data already exists
        cursor.execute(f"SELECT COUNT(*) FROM {STORES_TABLE}")
        if cursor.fetchone()[0] == 0:
            # Insert sample data
            _insert_sample_data(cursor)
        
        conn.commit()
        conn.close()
        logger.info(f'Sales database initialized in PostgreSQL.')
        
    except Exception as error:
        logger.error(f'Error initializing database: {error}')


def _insert_sample_data(cursor):
    """Insert sample data into all tables"""
    # Stores
    stores_data = [
        ('Downtown Store', 'Downtown', 'John Smith'),
        ('Mall Location', 'Shopping Mall', 'Jane Doe'),
        ('Airport Store', 'Airport Terminal', 'Bob Wilson')
    ]
    cursor.executemany(f"INSERT INTO {STORES_TABLE} (store_name, location, manager) VALUES (%s, %s, %s)", stores_data)
    
    # Categories
    categories_data = [
        ('Electronics', 'Electronic devices and accessories'),
        ('Clothing', 'Apparel and fashion items'),
        ('Books', 'Books and educational materials'),
        ('Home & Garden', 'Home improvement and garden supplies')
    ]
    cursor.executemany(f"INSERT INTO {CATEGORIES_TABLE} (category_name, description) VALUES (%s, %s)", categories_data)
    
    # Product Types
    product_types_data = [
        ('Premium', 'High-end products'),
        ('Standard', 'Regular quality products'),
        ('Budget', 'Economy products')
    ]
    cursor.executemany(f"INSERT INTO {PRODUCT_TYPES_TABLE} (type_name, description) VALUES (%s, %s)", product_types_data)
    
    # Products
    products_data = [
        ('Smartphone Pro', 1, 1, 899.99),
        ('Wireless Headphones', 1, 2, 149.99),
        ('Cotton T-Shirt', 2, 2, 24.99),
        ('Jeans', 2, 2, 79.99),
        ('Python Programming Guide', 3, 2, 49.99),
        ('Garden Hose', 4, 3, 29.99)
    ]
    cursor.executemany(f"INSERT INTO {PRODUCTS_TABLE} (product_name, category_id, type_id, price) VALUES (%s, %s, %s, %s)", products_data)
    
    # Customers
    customers_data = [
        ('Alice', 'Johnson', 'alice@email.com', '555-0101'),
        ('Bob', 'Smith', 'bob@email.com', '555-0102'),
        ('Carol', 'Davis', 'carol@email.com', '555-0103'),
        ('David', 'Wilson', 'david@email.com', '555-0104'),
        ('Eve', 'Brown', 'eve@email.com', '555-0105')
    ]
    cursor.executemany(f"INSERT INTO {CUSTOMERS_TABLE} (first_name, last_name, email, phone) VALUES (%s, %s, %s, %s)", customers_data)
    
    # Orders
    orders_data = [
        (1, 1, '2024-01-15', 1049.98),
        (2, 2, '2024-01-16', 104.98),
        (3, 1, '2024-02-01', 899.99),
        (4, 3, '2024-02-05', 154.97),
        (5, 2, '2024-02-10', 49.99)
    ]
    cursor.executemany(f"INSERT INTO {ORDERS_TABLE} (customer_id, store_id, order_date, total_amount) VALUES (%s, %s, %s, %s)", orders_data)
    
    # Order Items
    order_items_data = [
        (1, 1, 1, 899.99, 899.99),   # Alice: Smartphone
        (1, 2, 1, 149.99, 149.99),  # Alice: Headphones
        (2, 3, 2, 24.99, 49.98),    # Bob: 2 T-Shirts
        (2, 6, 1, 29.99, 29.99),    # Bob: Garden Hose
        (2, 3, 1, 24.99, 24.99),    # Bob: Another T-Shirt
        (3, 1, 1, 899.99, 899.99),  # Carol: Smartphone
        (4, 4, 1, 79.99, 79.99),    # David: Jeans
        (4, 3, 3, 24.99, 74.98),    # David: 3 T-Shirts
        (5, 5, 1, 49.99, 49.99)     # Eve: Book
    ]
    cursor.executemany(f"INSERT INTO {ORDER_ITEMS_TABLE} (order_id, product_id, quantity, unit_price, total_amount) VALUES (%s, %s, %s, %s, %s)", order_items_data)
    
    # Inventory
    inventory_data = [
        (1, 1, 50, 10),    # Smartphone at Downtown
        (1, 2, 25, 5),     # Smartphone at Mall
        (2, 1, 100, 20),   # Headphones at Downtown
        (3, 1, 200, 50),   # T-Shirts at Downtown
        (3, 2, 150, 30),   # T-Shirts at Mall
        (4, 2, 80, 15),    # Jeans at Mall
        (5, 1, 75, 15),    # Books at Downtown
        (6, 3, 40, 10)     # Garden Hose at Airport
    ]
    cursor.executemany(f"INSERT INTO {INVENTORY_TABLE} (product_id, store_id, quantity_in_stock, reorder_level) VALUES (%s, %s, %s, %s)", inventory_data)


# Utility functions
def test_connection() -> bool:
    """Test PostgreSQL connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


# Convenience functions
def execute_query(sql_query: str) -> str:
    """Execute SQL query and return JSON formatted results"""
    provider = PostgreSQLSchemaProvider()
    return provider.execute_query(sql_query)


def get_table_schema(table_name: str) -> Dict[str, Any]:
    """Get schema information for a table"""
    provider = PostgreSQLSchemaProvider()
    return provider.get_table_schema(table_name)


def get_all_tables() -> List[str]:
    """Get list of all tables in the database"""
    provider = PostgreSQLSchemaProvider()
    return provider.get_all_table_names()


def get_distinct_values(table_name: str, column_name: str) -> List[Any]:
    """Get distinct values from a column"""
    provider = PostgreSQLSchemaProvider()
    return provider.fetch_distinct_values(column_name, table_name)


def get_all_schemas() -> Dict[str, Dict[str, Any]]:
    """Get all table schemas"""
    provider = PostgreSQLSchemaProvider()
    return provider.get_all_schemas()


def get_table_metadata_string(table_name: str) -> str:
    """Return formatted schema metadata string for a single table."""
    provider = PostgreSQLSchemaProvider()
    return provider.get_table_metadata_string(table_name)


def get_table_metadata_from_list(table_names: List[str]) -> str:
    """Return formatted schema metadata strings for multiple tables."""
    provider = PostgreSQLSchemaProvider()
    return provider.get_table_metadata_from_list(table_names)


# Sample query functions for common operations
def get_sales_summary() -> str:
    """Get sales summary"""
    query = f"""
    SELECT 
        COUNT(DISTINCT o.order_id) as total_orders,
        COUNT(DISTINCT o.customer_id) as unique_customers,
        ROUND(SUM(oi.total_amount), 2) as total_revenue,
        ROUND(AVG(o.total_amount), 2) as avg_order_value
    FROM {ORDERS_TABLE} o
    JOIN {ORDER_ITEMS_TABLE} oi ON o.order_id = oi.order_id
    """
    return execute_query(query)


def get_top_products(limit: int = 5) -> str:
    """Get top selling products"""
    query = f"""
    SELECT 
        p.product_name,
        SUM(oi.quantity) as total_sold,
        ROUND(SUM(oi.total_amount), 2) as revenue
    FROM {PRODUCTS_TABLE} p
    JOIN {ORDER_ITEMS_TABLE} oi ON p.product_id = oi.product_id
    GROUP BY p.product_id, p.product_name
    ORDER BY total_sold DESC
    LIMIT {limit}
    """
    return execute_query(query)


def get_sales_by_store() -> str:
    """Get sales by store"""
    query = f"""
    SELECT 
        s.store_name,
        COUNT(o.order_id) as orders,
        ROUND(SUM(o.total_amount), 2) as revenue
    FROM {STORES_TABLE} s
    LEFT JOIN {ORDERS_TABLE} o ON s.store_id = o.store_id
    GROUP BY s.store_id, s.store_name
    ORDER BY revenue DESC
    """
    return execute_query(query)


def get_inventory_status() -> str:
    """Get current inventory status"""
    query = f"""
    SELECT 
        p.product_name,
        s.store_name,
        i.quantity_in_stock,
        i.reorder_level,
        CASE 
            WHEN i.quantity_in_stock <= i.reorder_level THEN 'LOW STOCK'
            ELSE 'OK'
        END as status
    FROM {INVENTORY_TABLE} i
    JOIN {PRODUCTS_TABLE} p ON i.product_id = p.product_id
    JOIN {STORES_TABLE} s ON i.store_id = s.store_id
    ORDER BY i.quantity_in_stock ASC
    """
    return execute_query(query)


# Initialize database when module is imported
try:
    init_db()
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")


if __name__ == "__main__":
    """Demo the database functionality"""
    print("🛍️  Sales Database Demo (PostgreSQL)")
    print("=" * 40)
    
    if not test_connection():
        print("❌ Error: Cannot connect to PostgreSQL")
        exit(1)
    
    print("\n📋 Available tables:")
    tables = get_all_tables()
    for table in tables:
        print(f"  - {table}")
    
    print("\n📊 Sales Summary:")
    print(get_sales_summary())
    
    print("\n🏆 Top Products:")
    print(get_top_products())
    
    print("\n🏪 Sales by Store:")
    print(get_sales_by_store())
    
    print("\n📦 Inventory Status:")
    print(get_inventory_status())
    
    print("\n🔍 Schema Example (products table):")
    schema = get_table_schema(PRODUCTS_TABLE)
    print(json.dumps(schema, indent=2))
    
    print("\n📋 Detailed Schema Information:")
    print(get_table_metadata_string(PRODUCTS_TABLE))