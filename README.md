# Video Ad Script Generator

A FastAPI application that generates creative ad scripts using LLM technology.

## Features

- Generate creative ad scripts based on campaign ideas
- Store generated scripts in a PostgreSQL database
- RESTful API for integration with other applications

## Installation

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -e .
   ```

3. Set up environment variables in a `.env` file:

   ```env
   GROQ_API_KEY=your_groq_api_key
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=your_db_name
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   ```

## Usage

### Running the API server

```bash
uvicorn app:app --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000)

### API Documentation

Once the server is running, you can access the interactive API documentation at:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Example API Request

```bash
curl -X 'POST' \
  'http://localhost:8000/generate-script' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "campaign_idea": "A refreshing new soda that makes you feel like you're floating in space"
}'
```
