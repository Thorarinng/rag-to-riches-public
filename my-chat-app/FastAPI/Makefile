# Makefile for FastAPI + Frontend RAG project

# Variables
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
APP := rag_api:app       # FastAPI app path relative to VENV dir
PORT := 8000
FRONTEND_DIR := ../      # Path to frontend (my-chat-app)

# Default target
.PHONY: all
all: help

# ----------------------------
# Setup Python environment
# ----------------------------
$(VENV)/bin/activate: requirements.txt
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo "Installing Python dependencies..."
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

.PHONY: setup
setup: $(VENV)/bin/activate
	@echo "✅ Python environment setup complete!"

# ----------------------------
# Run FastAPI + frontend
# ----------------------------
.PHONY: run
run: setup
	@echo "Installing frontend dependencies..."
	cd $(FRONTEND_DIR) && npm install
	@echo "Starting frontend..."
	cd $(FRONTEND_DIR) && npm run dev &
	@sleep 2
	@echo "Starting FastAPI server..."
	$(UVICORN) $(APP) --reload --port $(PORT)

# ----------------------------
# Activate Python environment
# ----------------------------
.PHONY: activate
activate:
	@echo "Activate virtual environment:"
	@echo "  macOS/Linux: source $(VENV)/bin/activate"
	@echo "  Windows CMD: $(VENV)\\Scripts\\activate.bat"
	@echo "  Windows PowerShell: $(VENV)\\Scripts\\Activate.ps1"

# ----------------------------
# Clean environment
# ----------------------------
.PHONY: clean
clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Done."

# ----------------------------
# Help
# ----------------------------
.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "  make setup      - Create Python venv and install dependencies"
	@echo "  make run        - Start frontend (npm) and FastAPI server"
	@echo "  make activate   - Show commands to activate Python venv"
	@echo "  make clean      - Remove virtual environment"
