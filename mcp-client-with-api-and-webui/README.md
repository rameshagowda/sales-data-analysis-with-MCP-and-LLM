cd mcp-client-with-api-and-webui/
uv venv
source .venv/bin/activate
cd FastAPI-with-OpenAI-and-MCP
uv init
uv add mcp openai python-dotenv fastapi psycopg2

# create .env and add OPENAI_API_KEY

touch .env

export OPENAI_API_KEY=

# run and test the API by calling endpoints

uv run uvicorn client:app --host 0.0.0.0 --port 8000 --reload
