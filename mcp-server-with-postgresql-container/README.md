# Setup Project

cd mcp-server-with-postgresql-container
uv init PostgresQL-Container
uv venv
source .venv/bin/activate

# Make sure postgresQL is running in docker container and # Store the connection string in .env

touch PostgresQL-Container/.env

# .env

DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=sales_production
DATABASE_USER=sales_user
DATABASE_PASSWORD=<your password>
DATABASE_URL=postgresql://sales_user:<your password>@localhost:5432/sales_production

# verify by login to db and querying for version

psql -h localhost -p 5432 -U sales_user -d sales_production

PYTHONPATH=PostgresQL-Container

# create src folder

touch PostgresQL-Container/config.py # check the content
touch PostgresQL-Container/test_connection.py # check the content
touch PostgresQL-Container/postgres.py

delete PostgresQL-Container/main.py

cd PostgresQL-Container

uv add mcp python-dotenv fastapi pydantic psycopg2-binary

# test the containerized postgres

uv run test_connection.py

# Setup database with dummy Sales data and Test - postgres.py already has data setup script and functions to pull data.

uv run postgres.py

# Now setup MCP server - resources, tools and prompts to database

cd FastMCP
uv init
delete main.py
touch mcp_server.py
uv add mcp python-dotenv fastapi psycopg2

uv run mcp_server.py

# Test the resources, tools and prompts

npx @modelcontextprotocol/inspector uv run mcp_server.py
