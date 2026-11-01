#!/bin/bash
set -e
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo "Add ANTHROPIC_API_KEY to .env, then run:"
echo "  uvicorn src.api.main:app --port 8000"
echo "  streamlit run app/streamlit_app.py"
