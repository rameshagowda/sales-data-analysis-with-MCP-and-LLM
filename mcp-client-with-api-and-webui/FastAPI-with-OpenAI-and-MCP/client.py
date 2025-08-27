# main.py
import asyncio
import json
import os
from contextlib import asynccontextmanager, AsyncExitStack
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import logging

# Add error handling for MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MCP not available: {e}")
    MCP_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables with defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found - OpenAI features will be disabled")

# ----------------------
# Pydantic models
# ----------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The message to send")
    use_tools: bool = Field(default=True, description="Whether to use available tools")

class ChatResponse(BaseModel):
    response: str
    tools_used: List[str] = []
    chart_data: Optional[Dict[str, Any]] = None

class ToolCallRequest(BaseModel):
    tool_name: str = Field(..., min_length=1)
    arguments: Dict[str, Any] = Field(default_factory=dict)

class ToolCallResponse(BaseModel):
    result: Any
    success: bool
    error: Optional[str] = None
    chart_data: Optional[Dict[str, Any]] = None

class SQLQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)

class SQLQueryResponse(BaseModel):
    result: Optional[str] = None
    success: bool
    error: Optional[str] = None
    chart_data: Optional[Dict[str, Any]] = None

class ChartData(BaseModel):
    chart_type: str = Field(..., pattern="^(pie|bar|line)$")
    data: List[Dict[str, Any]]
    title: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None

# ----------------------
# Chart Data Processing
# ----------------------
class ChartProcessor:
    @staticmethod
    def process_data_for_charts(data: Any, context: str = "") -> Optional[ChartData]:
        """Process raw data and determine the best chart type and format"""
        if not data:
            return None
            
        try:
            # Handle different data formats
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return None
                    
            if isinstance(data, dict):
                # Convert single dict to list
                if "summary" in data:
                    data = data["summary"]
                if isinstance(data, dict) and not any(isinstance(v, (list, dict)) for v in data.values()):
                    # Simple key-value pairs - good for pie chart
                    chart_data = [
                        {"name": str(k), "value": float(v)} 
                        for k, v in data.items() 
                        if isinstance(v, (int, float)) and v > 0
                    ]
                    if chart_data:
                        return ChartData(
                            chart_type="pie",
                            data=chart_data,
                            title=f"Distribution - {context}",
                        )
            
            if isinstance(data, list) and data:
                first_item = data[0]
                if isinstance(first_item, dict):
                    keys = list(first_item.keys())
                    
                    # Determine best chart type based on data structure
                    numeric_keys = [k for k in keys if isinstance(first_item.get(k), (int, float))]
                    string_keys = [k for k in keys if isinstance(first_item.get(k), str)]
                    
                    if len(numeric_keys) >= 1 and len(string_keys) >= 1:
                        # Good for bar chart
                        label_key = string_keys[0]
                        value_key = numeric_keys[0]
                        
                        chart_data = [
                            {"name": str(item[label_key]), "value": float(item[value_key])} 
                            for item in data 
                            if item.get(label_key) and isinstance(item.get(value_key), (int, float))
                        ]
                        
                        if chart_data:
                            chart_type = "pie" if len(chart_data) <= 10 else "bar"
                            return ChartData(
                                chart_type=chart_type,
                                data=chart_data,
                                title=f"{context} - {value_key.replace('_', ' ').title()}",
                                x_axis=label_key.replace('_', ' ').title(),
                                y_axis=value_key.replace('_', ' ').title()
                            )
            
        except Exception as e:
            logger.error(f"Error processing chart data: {e}")
            
        return None

# ----------------------
# MCP Client
# ----------------------
class SalesMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.tools = []
        self.resources = []
        self.prompts = []
        self.messages = []
        self.connected = False
        self.chart_processor = ChartProcessor()

    async def connect_to_server(self, server_script_path: str):
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP dependencies not available")
            
        if not server_script_path.endswith(".py"):
            raise ValueError("Server script must be a .py file")
        if not os.path.exists(server_script_path):
            raise FileNotFoundError(f"Server script not found: {server_script_path}")

        logger.info(f"Connecting to Sales MCP server: {server_script_path}")

        try:
            python_cmd = os.getenv("MCP_PYTHON_CMD", "python3")
            server_args = [server_script_path]

            server_params = StdioServerParameters(
                command=python_cmd,
                args=server_args,
                env=os.environ.copy()
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(stdio_transport[0], stdio_transport[1])
            )

            await self.session.initialize()
            await self._get_tools()
            await self._get_resources()
            await self._get_prompts()

            self.connected = True
            logger.info(f"Connected. Tools: {len(self.tools)}, Resources: {len(self.resources)}, Prompts: {len(self.prompts)}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            await self.cleanup()
            raise

    async def _get_tools(self):
        if not self.session:
            raise RuntimeError("Not connected")
        try:
            tools_result = await self.session.list_tools()
            self.tools = tools_result.tools
        except Exception as e:
            logger.error(f"Error getting tools: {e}")
            self.tools = []

    async def _get_resources(self):
        if not self.session:
            return
        try:
            res_result = await self.session.list_resources()
            self.resources = res_result.resources
        except Exception as e:
            logger.error(f"Error getting resources: {e}")
            self.resources = []

    async def _get_prompts(self):
        if not self.session:
            return
        try:
            prompts_result = await self.session.list_prompts()
            self.prompts = prompts_result.prompts
        except Exception as e:
            logger.error(f"Error getting prompts: {e}")
            self.prompts = []

    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any] = None):
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        if arguments is None:
            arguments = {}
            
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {[t.name for t in self.tools]}")
            
        try:
            result = await self.session.call_tool(tool_name, arguments)
            if result.isError:
                raise RuntimeError(f"Tool call failed: {result.content}")
            return result.content
        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            raise

    async def call_openai_llm(self, message: str, use_tools: bool = True):
        """Call OpenAI with optional MCP tool execution, returning structured JSON for charting."""
        if not self.llm:
            raise RuntimeError("OpenAI client not available - check OPENAI_API_KEY")
            
        logger.info(f"=== CHAT REQUEST ===")
        logger.info(f"Message: {message}")
        logger.info(f"Use tools: {use_tools}")
        logger.info(f"Available tools: {[t.name for t in self.tools]}")

        # Initialize system message if empty
        if not self.messages:
            self.messages.append({
                "role": "system",
                "content": """You are a sales assistant with access to a sales database via MCP tools.
When users ask about sales data, inventory, products, or business metrics, use the available tools to get real data.
Always provide helpful, accurate responses based on the actual data retrieved.
When providing data for charts, return a JSON object under 'tool_result' containing structured data."""
            })

        # Add user message
        self.messages.append({"role": "user", "content": message})
        tools_used = []
        tool_result = None
        chart_data = None

        # Prepare OpenAI call parameters
        call_params = {
            "model": "gpt-4o",
            "messages": self.messages,
            "temperature": 0.1
        }

        # Configure tools
        if use_tools and self.tools:
            tools_config = []
            for tool in self.tools:
                try:
                    schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
                    if not schema:
                        schema = {"type": "object", "properties": {}}

                    tools_config.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or f"Tool {tool.name}",
                            "parameters": schema
                        }
                    })
                except Exception as e:
                    logger.warning(f"Error configuring tool {tool.name}: {e}")

            if tools_config:
                call_params["tools"] = tools_config
                call_params["tool_choice"] = "auto"
                logger.info(f"Configured {len(tools_config)} tools")

        try:
            logger.info("Making initial OpenAI call...")
            response = await self.llm.chat.completions.create(**call_params)
            assistant_message = response.choices[0].message
            logger.info(f"Assistant response: {assistant_message.content}")
            logger.info(f"Tool calls: {len(assistant_message.tool_calls) if assistant_message.tool_calls else 0}")

            if assistant_message.tool_calls:
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args_str = tool_call.function.arguments or "{}"
                    try:
                        func_args = json.loads(func_args_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing tool arguments: {e}")
                        func_args = {}
                    
                    try:
                        raw_result = await self.call_mcp_tool(func_name, func_args)
                        tools_used.append(func_name)

                        # Save structured result for frontend charts
                        if isinstance(raw_result, (list, dict)):
                            tool_result = raw_result
                            # Process data for charting
                            chart_data = self.chart_processor.process_data_for_charts(
                                raw_result, 
                                context=func_name.replace('_', ' ').replace('tool', '').strip()
                            )

                        # Add tool output to message history
                        self.messages.append({
                            "role": "assistant",
                            "content": json.dumps(raw_result, indent=2, default=str) if isinstance(raw_result, (dict, list)) else str(raw_result)
                        })
                        
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        self.messages.append({
                            "role": "assistant",
                            "content": f"Tool {func_name} failed: {str(e)}"
                        })

                # Generate final assistant message after tool execution
                try:
                    final_response = await self.llm.chat.completions.create(
                        model="gpt-4o",
                        messages=self.messages,
                        temperature=0.1
                    )
                    final_content = final_response.choices[0].message.content or ""
                    self.messages.append({"role": "assistant", "content": final_content})
                    return final_content, tools_used, tool_result, chart_data
                except Exception as e:
                    logger.error(f"Final OpenAI call failed: {e}")
                    return "Error generating final response", tools_used, tool_result, chart_data

            else:
                # No tools called
                content = assistant_message.content or "No response generated"
                self.messages.append({"role": "assistant", "content": content})
                return content, tools_used, tool_result, chart_data

        except Exception as e:
            logger.error(f"OpenAI call failed: {str(e)}")
            error_msg = f"Error processing request: {str(e)}"
            self.messages.append({"role": "assistant", "content": error_msg})
            return error_msg, tools_used, tool_result, chart_data

    async def cleanup(self):
        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self.session = None
            self.connected = False

    def log_conversation(self):
        try:
            utils_dir = Path("utils")
            utils_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = utils_dir / f"conversation_{timestamp}.json"
            data = {"timestamp": datetime.now().isoformat(), "messages": self.messages}
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            return str(log_file)
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
            return None

# ----------------------
# FastAPI app with lifespan
# ----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Get server script path from environment
    server_script = os.getenv(
        "SALES_MCP_SERVER_SCRIPT", 
        "/Users/niam/Documents/develop/mcp and llm/sales-data-analysis-with-MCP-and-LLM/mcp-server-with-postgresql-container/FastMCP/mcp_server.py"  # Default to local file
    )
    
    # Convert to absolute path if relative
    if not os.path.isabs(server_script):
        server_script = os.path.abspath(server_script)
    
    logger.info(f"Starting FastAPI app")
    logger.info(f"MCP Available: {MCP_AVAILABLE}")
    logger.info(f"OpenAI Available: {OPENAI_API_KEY is not None}")
    logger.info(f"Server script: {server_script}")
    
    try:
        if MCP_AVAILABLE and os.path.exists(server_script):
            await sales_client.connect_to_server(server_script)
        else:
            logger.warning(f"MCP server not available or script not found: {server_script}")
        yield
    except Exception as e:
        logger.error(f"Error in lifespan startup: {e}")
        yield
    finally:
        await sales_client.cleanup()

app = FastAPI(
    title="Sales MCP Client API", 
    version="1.0.0", 
    description="FastAPI application for Sales Data Analysis with MCP integration",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sales_client = SalesMCPClient()

# ----------------------
# Health check endpoint
# ----------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mcp_connected": sales_client.connected,
        "mcp_available": MCP_AVAILABLE,
        "openai_available": OPENAI_API_KEY is not None,
        "tools_count": len(sales_client.tools),
        "timestamp": datetime.now().isoformat()
    }

# ----------------------
# Chat endpoint with chart support
# ----------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not sales_client.connected:
        raise HTTPException(status_code=503, detail="Not connected to MCP server")
    
    try:
        response, tools_used, tool_result, chart_data = await sales_client.call_openai_llm(
            request.message, 
            request.use_tools
        )
        
        chart_dict = None
        if chart_data:
            chart_dict = {
                "chart_type": chart_data.chart_type,
                "data": chart_data.data,
                "title": chart_data.title,
                "x_axis": chart_data.x_axis,
                "y_axis": chart_data.y_axis
            }
        
        return ChatResponse(
            response=response,
            tools_used=tools_used,
            chart_data=chart_dict
        )
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# ----------------------
# Tool endpoints with chart support
# ----------------------
@app.get("/tools")
async def get_tools():
    if not sales_client.connected:
        return {"tools": [], "message": "Not connected to MCP server"}
    
    return {
        "tools": [
            {
                "name": t.name, 
                "description": t.description, 
                "input_schema": t.inputSchema
            } for t in sales_client.tools
        ]
    }

@app.post("/tool/call", response_model=ToolCallResponse)
async def call_tool(request: ToolCallRequest):
    if not sales_client.connected:
        raise HTTPException(status_code=503, detail="Not connected to MCP server")
    
    try:
        result = await sales_client.call_mcp_tool(request.tool_name, request.arguments)
        
        # Process result for charting
        chart_data = sales_client.chart_processor.process_data_for_charts(
            result, 
            context=request.tool_name.replace('_', ' ').replace('tool', '').strip()
        )
        
        chart_dict = None
        if chart_data:
            chart_dict = {
                "chart_type": chart_data.chart_type,
                "data": chart_data.data,
                "title": chart_data.title,
                "x_axis": chart_data.x_axis,
                "y_axis": chart_data.y_axis
            }
        
        return ToolCallResponse(result=result, success=True, chart_data=chart_dict)
    except Exception as e:
        logger.error(f"Tool call error: {str(e)}")
        return ToolCallResponse(result=None, success=False, error=str(e))

# Helper function for tool calls with error handling
async def safe_tool_call(tool_name: str, arguments: Dict[str, Any] = None):
    if not sales_client.connected:
        raise HTTPException(status_code=503, detail="Not connected to MCP server")
    
    try:
        result = await sales_client.call_mcp_tool(tool_name, arguments or {})
        chart_data = sales_client.chart_processor.process_data_for_charts(result, tool_name)
        
        response = {tool_name.replace('get_', ''): result}
        if chart_data:
            response["chart_data"] = {
                "chart_type": chart_data.chart_type,
                "data": chart_data.data,
                "title": chart_data.title,
                "x_axis": chart_data.x_axis,
                "y_axis": chart_data.y_axis
            }
        
        return response
    except Exception as e:
        logger.error(f"Error in {tool_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting {tool_name}: {str(e)}")

# ----------------------
# Sales endpoints with chart support
# ----------------------
@app.get("/sales/summary")
async def sales_summary():
    return await safe_tool_call("get_sales_summary")

@app.get("/sales/top-products")
async def top_products(limit: int = 5):
    return await safe_tool_call("get_top_products", {"limit": limit})

@app.get("/sales/by-store")
async def sales_by_store():
    return await safe_tool_call("get_sales_by_store")

@app.get("/inventory/status")
async def inventory_status():
    return await safe_tool_call("get_inventory_status")

# ----------------------
# Table metadata endpoints
# ----------------------
@app.get("/tables/all")
async def all_tables():
    return await safe_tool_call("get_all_tables")

@app.get("/tables/schema")
async def table_schema(table_name: str):
    result = await safe_tool_call("get_table_schema", {"table_name": table_name})
    return {"table_name": table_name, "schema": result}

@app.get("/tables/distinct")
async def distinct_values(table_name: str, column_name: str):
    result = await safe_tool_call("get_distinct_values", {"table_name": table_name, "column_name": column_name})
    return {"table_name": table_name, "column_name": column_name, "values": result}

# ----------------------
# SQL endpoint with chart support
# ----------------------
@app.post("/sql", response_model=SQLQueryResponse)
async def execute_sql(request: SQLQueryRequest):
    if not sales_client.connected:
        raise HTTPException(status_code=503, detail="Not connected to MCP server")
    
    try:
        # Try the most common tool names for SQL execution
        possible_tool_names = ["execute_sql", "execute_sql_tool", "sql_query", "run_sql"]
        available_tools = [t.name for t in sales_client.tools]
        
        sql_tool_name = None
        for tool_name in possible_tool_names:
            if tool_name in available_tools:
                sql_tool_name = tool_name
                break
        
        if not sql_tool_name:
            logger.error(f"No SQL tool found. Available tools: {available_tools}")
            return SQLQueryResponse(
                result=None, 
                success=False, 
                error=f"No SQL execution tool found. Available tools: {available_tools}"
            )
        
        logger.info(f"Using SQL tool: {sql_tool_name}")
        result = await sales_client.call_mcp_tool(sql_tool_name, {"query": request.query})
        
        # Process result for charting
        chart_data = sales_client.chart_processor.process_data_for_charts(result, "SQL Query Result")
        
        chart_dict = None
        if chart_data:
            chart_dict = {
                "chart_type": chart_data.chart_type,
                "data": chart_data.data,
                "title": chart_data.title,
                "x_axis": chart_data.x_axis,
                "y_axis": chart_data.y_axis
            }
        
        return SQLQueryResponse(
            result=json.dumps(result, indent=2, default=str) if isinstance(result, (dict, list)) else str(result),
            success=True,
            chart_data=chart_dict
        )
        
    except Exception as e:
        logger.error(f"SQL execution failed: {str(e)}")
        return SQLQueryResponse(
            result=None,
            success=False,
            error=str(e)
        )

# ----------------------
# Chart visualization endpoint
# ----------------------
@app.get("/chart/demo", response_class=HTMLResponse)
async def chart_demo():
    """Demo endpoint showing how to render charts from the API data"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sales Charts Demo</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .chart-container { margin: 20px 0; width: 100%; height: 400px; }
            .controls { margin: 20px 0; }
            button { margin: 5px; padding: 10px 20px; cursor: pointer; }
            #status { margin: 10px 0; padding: 10px; background: #f0f0f0; }
            .error { background: #fee; color: #c00; }
            .success { background: #efe; color: #060; }
        </style>
    </head>
    <body>
        <h1>Sales Dashboard</h1>
        <div class="controls">
            <button onclick="checkHealth()">Check Health</button>
            <button onclick="loadSalesSummary()">Sales Summary</button>
            <button onclick="loadTopProducts()">Top Products</button>
            <button onclick="loadSalesByStore()">Sales by Store</button>
            <button onclick="loadInventoryStatus()">Inventory Status</button>
        </div>
        <div id="status">Click a button to load data and generate charts</div>
        <div class="chart-container">
            <canvas id="myChart"></canvas>
        </div>

        <script>
            let currentChart = null;

            async function checkHealth() {
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = 'Checking health...';
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    statusDiv.className = data.mcp_connected ? 'success' : 'error';
                    statusDiv.textContent = `Health: ${JSON.stringify(data, null, 2)}`;
                } catch (error) {
                    statusDiv.className = 'error';
                    statusDiv.textContent = `Health check failed: ${error.message}`;
                }
            }

            async function loadData(endpoint) {
                const statusDiv = document.getElementById('status');
                statusDiv.className = '';
                statusDiv.textContent = `Loading ${endpoint}...`;
                
                try {
                    const response = await fetch(endpoint);
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    
                    if (data.chart_data) {
                        renderChart(data.chart_data);
                        statusDiv.className = 'success';
                        statusDiv.textContent = `Loaded ${endpoint} successfully`;
                    } else {
                        statusDiv.className = '';
                        statusDiv.textContent = `No chart data available for ${endpoint}`;
                        console.log('Raw data:', data);
                    }
                } catch (error) {
                    statusDiv.className = 'error';
                    statusDiv.textContent = `Error loading ${endpoint}: ${error.message}`;
                    console.error('Error:', error);
                }
            }

            function renderChart(chartData) {
                const ctx = document.getElementById('myChart').getContext('2d');
                
                if (currentChart) {
                    currentChart.destroy();
                }

                const config = {
                    type: chartData.chart_type,
                    data: {
                        labels: chartData.data.map(item => item.name),
                        datasets: [{
                            label: chartData.y_axis || 'Value',
                            data: chartData.data.map(item => item.value),
                            backgroundColor: [
                                'rgba(255, 99, 132, 0.8)',
                                'rgba(54, 162, 235, 0.8)',
                                'rgba(255, 205, 86, 0.8)',
                                'rgba(75, 192, 192, 0.8)',
                                'rgba(153, 102, 255, 0.8)',
                                'rgba(255, 159, 64, 0.8)',
                                'rgba(199, 199, 199, 0.8)',
                                'rgba(83, 102, 255, 0.8)'
                            ],
                            borderColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 205, 86, 1)',
                                'rgba(75, 192, 192, 1)',
                                'rgba(153, 102, 255, 1)',
                                'rgba(255, 159, 64, 1)',
                                'rgba(199, 199, 199, 1)',
                                'rgba(83, 102, 255, 1)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: chartData.title
                            },
                            legend: {
                                display: chartData.chart_type === 'pie'
                            }
                        },
                        scales: chartData.chart_type !== 'pie' ? {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: chartData.y_axis || 'Value'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: chartData.x_axis || 'Category'
                                }
                            }
                        } : {}
                    }
                };

                currentChart = new Chart(ctx, config);
            }

            function loadSalesSummary() { loadData('/sales/summary'); }
            function loadTopProducts() { loadData('/sales/top-products'); }
            function loadSalesByStore() { loadData('/sales/by-store'); }
            function loadInventoryStatus() { loadData('/inventory/status'); }
        </script>
    </body>
    </html>
    """

# ----------------------
# Debug endpoints
# ----------------------
@app.get("/debug/tools")
async def debug_tools():
    """Debug endpoint to check tool availability"""
    return {
        "connected": sales_client.connected,
        "mcp_available": MCP_AVAILABLE,
        "openai_available": OPENAI_API_KEY is not None,
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "schema": t.inputSchema
            } for t in sales_client.tools
        ],
        "resources": len(sales_client.resources),
        "prompts": len(sales_client.prompts)
    }

@app.get("/debug/conversation")
async def debug_conversation():
    """Get current conversation messages"""
    return {"messages": sales_client.messages}

@app.post("/debug/log-conversation")
async def log_conversation():
    """Save current conversation to file"""
    log_file = sales_client.log_conversation()
    return {"log_file": log_file, "message_count": len(sales_client.messages)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )