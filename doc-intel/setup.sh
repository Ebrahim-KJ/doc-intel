#!/bin/bash
echo "🔍 Setting up DocIntel..."

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
echo "⚠️  Add your ANTHROPIC_API_KEY to .env before running"

echo "✅ Setup complete!"
echo ""
echo "To run:"
echo "  Terminal 1: uvicorn src.api.main:app --reload --port 8000"
echo "  Terminal 2: streamlit run app/streamlit_app.py"
echo ""
echo "Or with Docker:"
echo "  docker-compose up --build"
