import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from typing import Dict, Any, Optional, List
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Sales MCP Chat Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        
    def check_health(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def get_tools(self) -> Dict[str, Any]:
        """Get available MCP tools"""
        try:
            response = requests.get(f"{self.base_url}/tools", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"tools": [], "error": str(e)}
    
    def chat(self, message: str, use_tools: bool = True) -> Dict[str, Any]:
        """Send chat message to API"""
        try:
            payload = {
                "message": message,
                "use_tools": use_tools
            }
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"response": f"Error: {str(e)}", "tools_used": [], "chart_data": None}
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a specific MCP tool"""
        try:
            payload = {
                "tool_name": tool_name,
                "arguments": arguments or {}
            }
            response = requests.post(
                f"{self.base_url}/tool/call",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"result": None, "success": False, "error": str(e)}
    
    def execute_sql(self, query: str) -> Dict[str, Any]:
        """Execute SQL query"""
        try:
            payload = {"query": query}
            response = requests.post(
                f"{self.base_url}/sql",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"result": None, "success": False, "error": str(e)}

def create_chart(chart_data: Dict[str, Any]) -> go.Figure:
    """Create Plotly chart from API chart data"""
    if not chart_data or not chart_data.get('data'):
        return None
    
    chart_type = chart_data.get('chart_type', 'bar')
    data = chart_data.get('data', [])
    title = chart_data.get('title', 'Chart')
    x_axis = chart_data.get('x_axis', 'Category')
    y_axis = chart_data.get('y_axis', 'Value')
    
    # Convert data to DataFrame for easier handling
    df = pd.DataFrame(data)
    
    if chart_type == 'pie':
        fig = px.pie(
            df, 
            values='value', 
            names='name',
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
    elif chart_type == 'bar':
        fig = px.bar(
            df,
            x='name',
            y='value',
            title=title,
            labels={'name': x_axis, 'value': y_axis},
            color='value',
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        
    elif chart_type == 'line':
        fig = px.line(
            df,
            x='name',
            y='value',
            title=title,
            labels={'name': x_axis, 'value': y_axis},
            markers=True
        )
        
    else:  # Default to bar
        fig = px.bar(
            df,
            x='name',
            y='value',
            title=title,
            labels={'name': x_axis, 'value': y_axis}
        )
    
    # Update layout for better appearance
    fig.update_layout(
        margin=dict(t=50, b=50, l=50, r=50),
        height=400,
        font=dict(size=12)
    )
    
    return fig

def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with optional chart"""
    with st.chat_message("user" if is_user else "assistant"):
        if is_user:
            st.write(message.get('content', ''))
        else:
            # Display assistant response
            st.write(message.get('response', ''))
            
            # Show tools used
            tools_used = message.get('tools_used', [])
            if tools_used:
                with st.expander(f"🔧 Tools Used: {', '.join(tools_used)}", expanded=False):
                    st.write("The assistant used the following MCP tools to get this information:")
                    for tool in tools_used:
                        st.write(f"- `{tool}`")
            
            # Display chart if available
            chart_data = message.get('chart_data')
            if chart_data:
                try:
                    fig = create_chart(chart_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient(API_BASE_URL)
    if 'health_status' not in st.session_state:
        st.session_state.health_status = None
    if 'available_tools' not in st.session_state:
        st.session_state.available_tools = []

def main():
    initialize_session_state()
    
    st.title("💬 Sales MCP Chat Assistant")
    st.markdown("Chat with your sales data using MCP tools and AI assistance!")
    
    # Sidebar for controls and status
    with st.sidebar:
        st.header("🔧 Controls")
        
        # API Status
        st.subheader("📊 API Status")
        
        if st.button("🔄 Check API Health", use_container_width=True):
            with st.spinner("Checking API health..."):
                st.session_state.health_status = st.session_state.api_client.check_health()
        
        if st.session_state.health_status:
            status = st.session_state.health_status
            if status.get('status') == 'healthy':
                st.success("✅ API Connected")
                st.json({
                    "MCP Connected": status.get('mcp_connected', False),
                    "Tools Available": status.get('tools_count', 0),
                    "OpenAI Available": status.get('openai_available', False)
                })
            else:
                st.error("❌ API Error")
                st.error(status.get('message', 'Unknown error'))
        
        # Tool Management
        st.subheader("🛠️ Available Tools")
        
        if st.button("📋 Load Tools", use_container_width=True):
            with st.spinner("Loading tools..."):
                tools_response = st.session_state.api_client.get_tools()
                st.session_state.available_tools = tools_response.get('tools', [])
        
        if st.session_state.available_tools:
            st.success(f"Found {len(st.session_state.available_tools)} tools")
            with st.expander("View Tools", expanded=False):
                for tool in st.session_state.available_tools:
                    st.write(f"**{tool['name']}**")
                    st.write(f"_{tool.get('description', 'No description')}_")
                    st.divider()
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        quick_actions = {
            "📈 Sales Summary": "Show me a sales summary with charts",
            "🏆 Top Products": "What are the top selling products?",
            "🏪 Sales by Store": "Show sales performance by store",
            "📦 Inventory Status": "What's the current inventory status?",
            "💰 Revenue Analysis": "Analyze our revenue trends",
            "📊 Custom SQL": "I want to write a custom SQL query"
        }
        
        for action_name, action_message in quick_actions.items():
            if st.button(action_name, use_container_width=True):
                # Add the quick action message to chat
                st.session_state.messages.append({
                    'content': action_message,
                    'is_user': True,
                    'timestamp': datetime.now()
                })
                st.rerun()
        
        # Settings
        st.subheader("⚙️ Settings")
        use_tools = st.toggle("Use MCP Tools", value=True, help="Enable/disable MCP tool usage")
        
        # Clear Chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(message, message.get('is_user', False))
        
        # Chat input
        if prompt := st.chat_input("Ask about your sales data..."):
            # Add user message
            user_message = {
                'content': prompt,
                'is_user': True,
                'timestamp': datetime.now()
            }
            st.session_state.messages.append(user_message)
            
            # Display user message immediately
            display_chat_message(user_message, True)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.api_client.chat(prompt, use_tools)
                    
                    # Add assistant message
                    assistant_message = {
                        'response': response.get('response', ''),
                        'tools_used': response.get('tools_used', []),
                        'chart_data': response.get('chart_data'),
                        'is_user': False,
                        'timestamp': datetime.now()
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Display the response
                    st.write(assistant_message['response'])
                    
                    # Show tools used
                    if assistant_message['tools_used']:
                        with st.expander(f"🔧 Tools Used: {', '.join(assistant_message['tools_used'])}", expanded=False):
                            st.write("The assistant used the following MCP tools:")
                            for tool in assistant_message['tools_used']:
                                st.write(f"- `{tool}`")
                    
                    # Display chart if available
                    if assistant_message['chart_data']:
                        try:
                            fig = create_chart(assistant_message['chart_data'])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating chart: {str(e)}")

    # Additional features in tabs
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Direct Tool Calls", "📝 SQL Query", "📊 Dashboard"])
    
    with tab1:
        st.subheader("Direct MCP Tool Execution")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.session_state.available_tools:
                selected_tool = st.selectbox(
                    "Select Tool",
                    options=[tool['name'] for tool in st.session_state.available_tools],
                    help="Choose an MCP tool to execute directly"
                )
                
                # Show tool description
                if selected_tool:
                    tool_info = next(t for t in st.session_state.available_tools if t['name'] == selected_tool)
                    st.info(f"**Description:** {tool_info.get('description', 'No description')}")
                
                # Arguments input (simple JSON)
                arguments_text = st.text_area(
                    "Arguments (JSON)",
                    value="{}",
                    help="Enter tool arguments as JSON"
                )
                
                if st.button("🚀 Execute Tool"):
                    try:
                        arguments = json.loads(arguments_text) if arguments_text.strip() else {}
                        
                        with st.spinner(f"Executing {selected_tool}..."):
                            result = st.session_state.api_client.call_tool(selected_tool, arguments)
                            
                        if result.get('success'):
                            st.success("✅ Tool executed successfully!")
                            
                            with col2:
                                st.subheader("Result")
                                st.json(result.get('result'))
                                
                                # Show chart if available
                                if result.get('chart_data'):
                                    fig = create_chart(result['chart_data'])
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"❌ Tool execution failed: {result.get('error')}")
                            
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON in arguments")
            else:
                st.warning("No tools available. Click 'Load Tools' in the sidebar first.")
    
    with tab2:
        st.subheader("Custom SQL Query")
        
        sql_query = st.text_area(
            "SQL Query",
            value="SELECT * FROM sales LIMIT 10;",
            height=100,
            help="Enter your SQL query here"
        )
        
        if st.button("🔍 Execute SQL"):
            if sql_query.strip():
                with st.spinner("Executing SQL query..."):
                    result = st.session_state.api_client.execute_sql(sql_query)
                
                if result.get('success'):
                    st.success("✅ Query executed successfully!")
                    
                    # Display result
                    if result.get('result'):
                        try:
                            # Try to parse as JSON and display as dataframe
                            parsed_result = json.loads(result['result'])
                            if isinstance(parsed_result, list) and parsed_result:
                                df = pd.DataFrame(parsed_result)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.json(parsed_result)
                        except:
                            st.text(result['result'])
                    
                    # Show chart if available
                    if result.get('chart_data'):
                        fig = create_chart(result['chart_data'])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ SQL execution failed: {result.get('error')}")
            else:
                st.warning("Please enter a SQL query")
    
    with tab3:
        st.subheader("Sales Dashboard")
        st.markdown("Quick overview of key metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Load Sales Summary", use_container_width=True):
                with st.spinner("Loading sales summary..."):
                    try:
                        response = requests.get(f"{API_BASE_URL}/sales/summary", timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('chart_data'):
                                fig = create_chart(data['chart_data'])
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            st.json(data.get('summary', {}))
                        else:
                            st.error("Failed to load sales summary")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("🏆 Load Top Products", use_container_width=True):
                with st.spinner("Loading top products..."):
                    try:
                        response = requests.get(f"{API_BASE_URL}/sales/top-products", timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('chart_data'):
                                fig = create_chart(data['chart_data'])
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            st.json(data.get('top_products', []))
                        else:
                            st.error("Failed to load top products")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()