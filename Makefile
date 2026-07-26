.PHONY: install test lint demo api ui

install:
	python -m pip install -e ".[ui,dev]"

test:
	python -m pytest

lint:
	python -m ruff check .

demo:
	python -m biorag.cli demo

api:
	uvicorn biorag.api:app --reload

ui:
	streamlit run app.py

