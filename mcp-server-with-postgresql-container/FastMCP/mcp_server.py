#!/usr/bin/env python3
"""
Sales Database MCP Server
A simple MCP server for the PostgreSQL sales database tool.
"""

import json
import sys
from typing import Any, Dict, List
import logging
import os


import mcp
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PostgresQL-Container'))

# Import your existing database module
from postgres import (
    PostgreSQLSchemaProvider,
    execute_query,
    get_table_schema,
    get_all_tables,
    get_distinct_values,
    get_all_schemas,
    get_table_metadata_string,
    get_table_metadata_from_list,
    get_sales_summary,
    get_top_products,
    get_sales_by_store,
    get_inventory_status,
    test_connection
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sales_mcp_server")

class SalesMCPServer:
    """Simple MCP server for sales database operations."""
    
    def __init__(self):
        self.provider = PostgreSQLSchemaProvider()
        
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            # Handle notification messages (no id field)
            if request_id is None and method in ["notifications/initialized"]:
                return None  # Don't respond to notifications
            
            if method == "initialize":
                return self._initialize(request_id)
            elif method == "tools/list":
                return self._list_tools(request_id)
            elif method == "tools/call":
                return self._call_tool(params, request_id)
            elif method == "resources/list":
                return self._list_resources(request_id)
            elif method == "resources/read":
                return self._read_resource(params, request_id)
            elif method == "prompts/list":
                return self._list_prompts(request_id)
            elif method == "prompts/get":
                return self._get_prompt(params, request_id)
            else:
                return self._error(f"Unknown method: {method}", request_id)
                
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return self._error(str(e), request.get("id"))
    
    def _initialize(self, request_id: str = None) -> Dict[str, Any]:
        """Initialize the MCP server."""
        response = {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": {
                    "name": "sales-database",
                    "version": "1.0.0"
                }
            }
        }
        if request_id is not None:
            response["id"] = request_id
        return response
    
    def _list_tools(self, request_id: str = None) -> Dict[str, Any]:
        """List available tools."""
        tools = [
            {
                "name": "execute_sql",
                "description": "Execute a SQL query against the sales database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_table_schema",
                "description": "Get schema information for a specific table",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table"
                        }
                    },
                    "required": ["table_name"]
                }
            },
            {
                "name": "get_all_tables",
                "description": "List all tables in the database",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_table_metadata",
                "description": "Get formatted metadata for one or more tables",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of table names to get metadata for"
                        }
                    },
                    "required": ["table_names"]
                }
            },
            {
                "name": "get_distinct_values",
                "description": "Get distinct values from a column",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table"
                        },
                        "column_name": {
                            "type": "string",
                            "description": "Name of the column"
                        }
                    },
                    "required": ["table_name", "column_name"]
                }
            },
            {
                "name": "get_sales_summary",
                "description": "Get overall sales summary statistics",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_top_products",
                "description": "Get top selling products",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of top products to return",
                            "default": 5
                        }
                    }
                }
            },
            {
                "name": "get_sales_by_store",
                "description": "Get sales statistics by store",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_inventory_status",
                "description": "Get current inventory status",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "test_connection",
                "description": "Test database connection",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        
        response = {
            "jsonrpc": "2.0",
            "result": {
                "tools": tools
            }
        }
        if request_id is not None:
            response["id"] = request_id
        return response
    
    def _call_tool(self, params: Dict[str, Any], request_id: str = None) -> Dict[str, Any]:
        """Call a tool with the given parameters."""
        try:
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "execute_sql":
                query = arguments.get("query")
                result = execute_query(query)
                return self._success([{"type": "text", "text": result}], request_id)
            
            elif tool_name == "get_table_schema":
                table_name = arguments.get("table_name")
                result = get_table_schema(table_name)
                return self._success([{"type": "text", "text": json.dumps(result, indent=2)}], request_id)
            
            elif tool_name == "get_all_tables":
                result = get_all_tables()
                return self._success([{"type": "text", "text": json.dumps(result, indent=2)}], request_id)
            
            elif tool_name == "get_table_metadata":
                table_names = arguments.get("table_names", [])
                result = get_table_metadata_from_list(table_names)
                return self._success([{"type": "text", "text": result}], request_id)
            
            elif tool_name == "get_distinct_values":
                table_name = arguments.get("table_name")
                column_name = arguments.get("column_name")
                result = get_distinct_values(table_name, column_name)
                return self._success([{"type": "text", "text": json.dumps(result, indent=2)}], request_id)
            
            elif tool_name == "get_sales_summary":
                result = get_sales_summary()
                return self._success([{"type": "text", "text": result}], request_id)
            
            elif tool_name == "get_top_products":
                limit = arguments.get("limit", 5)
                result = get_top_products(limit)
                return self._success([{"type": "text", "text": result}], request_id)
            
            elif tool_name == "get_sales_by_store":
                result = get_sales_by_store()
                return self._success([{"type": "text", "text": result}], request_id)
            
            elif tool_name == "get_inventory_status":
                result = get_inventory_status()
                return self._success([{"type": "text", "text": result}], request_id)
            
            elif tool_name == "test_connection":
                result = test_connection()
                status = "Connection successful" if result else "Connection failed"
                return self._success([{"type": "text", "text": status}], request_id)
            
            else:
                return self._error(f"Unknown tool: {tool_name}", request_id)
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return self._error(str(e), request_id)
    
    def _list_resources(self, request_id: str = None) -> Dict[str, Any]:
        """List available resources."""
        resources = [
            {
                "uri": "sales://database/schema",
                "name": "Database Schema",
                "description": "Complete database schema information",
                "mimeType": "application/json"
            },
            {
                "uri": "sales://database/tables",
                "name": "Database Tables",
                "description": "List of all database tables",
                "mimeType": "application/json"
            }
        ]
        
        response = {
            "jsonrpc": "2.0",
            "result": {
                "resources": resources
            }
        }
        if request_id is not None:
            response["id"] = request_id
        return response
    
    def _read_resource(self, params: Dict[str, Any], request_id: str = None) -> Dict[str, Any]:
        """Read a resource."""
        try:
            uri = params.get("uri")
            
            if uri == "sales://database/schema":
                schemas = get_all_schemas()
                content = json.dumps(schemas, indent=2)
                response = {
                    "jsonrpc": "2.0",
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": content
                            }
                        ]
                    }
                }
                if request_id is not None:
                    response["id"] = request_id
                return response
            
            elif uri == "sales://database/tables":
                tables = get_all_tables()
                content = json.dumps(tables, indent=2)
                response = {
                    "jsonrpc": "2.0",
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": content
                            }
                        ]
                    }
                }
                if request_id is not None:
                    response["id"] = request_id
                return response
            
            else:
                return self._error(f"Unknown resource: {uri}", request_id)
                
        except Exception as e:
            logger.error(f"Error reading resource: {e}")
            return self._error(str(e), request_id)
    
    def _list_prompts(self, request_id: str = None) -> Dict[str, Any]:
        """List available prompts."""
        prompts = [
            {
                "name": "analyze_sales_performance",
                "description": "Analyze sales performance with comprehensive insights",
                "arguments": [
                    {
                        "name": "time_period",
                        "description": "Time period to analyze (e.g., 'last month', 'Q1 2024', 'all time')",
                        "required": False
                    },
                    {
                        "name": "store_focus",
                        "description": "Specific store to focus on (optional)",
                        "required": False
                    }
                ]
            },
            {
                "name": "product_insights",
                "description": "Generate insights about product performance and inventory",
                "arguments": [
                    {
                        "name": "category",
                        "description": "Product category to focus on (optional)",
                        "required": False
                    },
                    {
                        "name": "analysis_type",
                        "description": "Type of analysis (sales, inventory, trends)",
                        "required": False
                    }
                ]
            },
            {
                "name": "customer_analysis",
                "description": "Analyze customer behavior and purchasing patterns",
                "arguments": [
                    {
                        "name": "customer_segment",
                        "description": "Customer segment to analyze (optional)",
                        "required": False
                    }
                ]
            },
            {
                "name": "inventory_report",
                "description": "Generate comprehensive inventory status report",
                "arguments": [
                    {
                        "name": "alert_level",
                        "description": "Focus on specific alert level (low_stock, all)",
                        "required": False
                    }
                ]
            },
            {
                "name": "sql_query_helper",
                "description": "Help construct SQL queries for the sales database",
                "arguments": [
                    {
                        "name": "query_goal",
                        "description": "What you want to find out from the database",
                        "required": True
                    },
                    {
                        "name": "complexity",
                        "description": "Query complexity level (simple, intermediate, complex)",
                        "required": False
                    }
                ]
            }
        ]
        
        response = {
            "jsonrpc": "2.0",
            "result": {
                "prompts": prompts
            }
        }
        if request_id is not None:
            response["id"] = request_id
        return response
    
    def _get_prompt(self, params: Dict[str, Any], request_id: str = None) -> Dict[str, Any]:
        """Get a specific prompt with arguments filled in."""
        try:
            prompt_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if prompt_name == "analyze_sales_performance":
                time_period = arguments.get("time_period", "all available data")
                store_focus = arguments.get("store_focus", "all stores")
                
                prompt_text = f"""You are a sales analyst working with a PostgreSQL sales database. Analyze sales performance for {time_period} focusing on {store_focus}.

Please provide a comprehensive sales analysis including:

1. **Overall Performance Metrics**
   - Total revenue, orders, and customers
   - Average order value
   - Growth trends (if time period allows)

2. **Store Performance**
   - Revenue and order count by store
   - Best and worst performing locations
   - Store-specific insights

3. **Product Performance**
   - Top selling products by quantity and revenue
   - Category performance breakdown
   - Product trends and insights

4. **Customer Insights**
   - Customer acquisition and retention
   - Purchase patterns
   - Customer value analysis

Use the available database tools to gather data and provide actionable insights with specific recommendations."""

            elif prompt_name == "product_insights":
                category = arguments.get("category", "all categories")
                analysis_type = arguments.get("analysis_type", "comprehensive")
                
                prompt_text = f"""You are a product manager analyzing {category} for {analysis_type} insights using the sales database.

Please provide detailed product analysis including:

1. **Sales Performance**
   - Best and worst selling products
   - Revenue contribution by product
   - Sales velocity and trends

2. **Inventory Analysis**
   - Current stock levels
   - Products needing reorder
   - Inventory turnover insights

3. **Category Analysis** (if analyzing multiple categories)
   - Performance comparison across categories
   - Category growth trends
   - Cross-category insights

4. **Recommendations**
   - Products to promote or discontinue
   - Inventory optimization suggestions
   - Pricing and positioning recommendations

Use SQL queries to gather comprehensive data and provide actionable business insights."""

            elif prompt_name == "customer_analysis":
                segment = arguments.get("customer_segment", "all customers")
                
                prompt_text = f"""You are a customer success manager analyzing {segment} behavior using the sales database.

Please provide comprehensive customer analysis including:

1. **Customer Overview**
   - Total customer count and demographics
   - New vs returning customer ratios
   - Customer distribution across stores

2. **Purchase Behavior**
   - Average order values and frequencies
   - Most popular products per customer segment
   - Seasonal or time-based patterns

3. **Customer Value Analysis**
   - High-value customers identification
   - Customer lifetime value insights
   - Purchase frequency patterns

4. **Retention and Growth**
   - Customer retention indicators
   - Opportunities for upselling/cross-selling
   - Customer satisfaction insights (based on purchase data)

5. **Actionable Recommendations**
   - Customer retention strategies
   - Personalization opportunities
   - Marketing campaign suggestions

Use database queries to support all insights with concrete data."""

            elif prompt_name == "inventory_report":
                alert_level = arguments.get("alert_level", "comprehensive")
                
                prompt_text = f"""You are an inventory manager creating a {alert_level} inventory status report using the sales database.

Please provide detailed inventory analysis including:

1. **Current Stock Status**
   - Overall inventory levels across all stores
   - Products at or below reorder levels
   - Overstocked items identification

2. **Store-by-Store Analysis**
   - Inventory distribution across locations
   - Store-specific stock issues
   - Inter-store transfer opportunities

3. **Product Category Insights**
   - Category-level inventory health
   - Fast-moving vs slow-moving products
   - Seasonal inventory considerations

4. **Financial Impact**
   - Value of current inventory
   - Potential lost sales from stockouts
   - Carrying costs of excess inventory

5. **Action Items**
   - Immediate reorder recommendations
   - Inventory redistribution suggestions
   - Process improvement opportunities

Generate specific, actionable recommendations with supporting data from SQL queries."""

            elif prompt_name == "sql_query_helper":
                query_goal = arguments.get("query_goal")
                complexity = arguments.get("complexity", "intermediate")
                
                prompt_text = f"""You are a SQL expert helping to construct {complexity} queries for a PostgreSQL sales database.

**Query Goal:** {query_goal}

**Database Schema Available:**
- customers (customer_id, first_name, last_name, email, phone)
- products (product_id, product_name, category_id, type_id, price)
- orders (order_id, customer_id, store_id, order_date, total_amount)
- order_items (order_item_id, order_id, product_id, quantity, unit_price, total_amount)
- stores (store_id, store_name, location, manager)
- categories (category_id, category_name, description)
- product_types (type_id, type_name, description)
- inventory (inventory_id, product_id, store_id, quantity_in_stock, reorder_level)

Please help construct the SQL query by:

1. **Understanding the Requirements**
   - Break down what data is needed
   - Identify required tables and relationships

2. **Query Construction**
   - Start with basic SELECT structure
   - Add appropriate JOINs
   - Include necessary WHERE clauses
   - Add GROUP BY, ORDER BY as needed

3. **Query Optimization**
   - Suggest indexes if relevant
   - Optimize for performance
   - Handle edge cases

4. **Result Interpretation**
   - Explain what the results mean
   - Suggest follow-up queries if helpful

First, use the get_table_metadata tool to examine relevant table structures, then construct and test the query."""

            else:
                return self._error(f"Unknown prompt: {prompt_name}", request_id)
            
            response = {
                "jsonrpc": "2.0",
                "result": {
                    "description": f"Generated prompt for {prompt_name}",
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": prompt_text
                            }
                        }
                    ]
                }
            }
            if request_id is not None:
                response["id"] = request_id
            return response
            
        except Exception as e:
            logger.error(f"Error getting prompt: {e}")
            return self._error(str(e), request_id)
    
    def _success(self, content: List[Dict[str, Any]], request_id: str = None) -> Dict[str, Any]:
        """Create a success response."""
        response = {
            "jsonrpc": "2.0",
            "result": {
                "content": content
            }
        }
        if request_id:
            response["id"] = request_id
        return response
    
    def _error(self, message: str, request_id: str = None) -> Dict[str, Any]:
        """Create an error response."""
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": -1,
                "message": message
            }
        }
        if request_id:
            response["id"] = request_id
        return response

def main():
    """Main server loop."""
    server = SalesMCPServer()
    
    logger.info("Sales Database MCP Server starting...")
    
    # Test database connection on startup
    if not test_connection():
        logger.error("Failed to connect to database. Please check your connection settings.")
        sys.exit(1)
    
    logger.info("Database connection successful. Server ready.")
    
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
                response = server.handle_request(request)
                
                # Only send response if it's not None (for notifications)
                if response is not None:
                    print(json.dumps(response))
                    sys.stdout.flush()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "Internal error"
                    }
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
    
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     mcp.run(transport="streamable-http")