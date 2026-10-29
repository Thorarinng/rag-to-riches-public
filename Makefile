# -----------------------------
# Makefile for Python project setup
# -----------------------------

# Name of virtual environment folder
VENV := venv

# Python used to create the venv (override with: make setup PY=python3.12)
PY ?= python3

# Path to pip inside venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make setup     - Create venv, install dependencies, and create .env"
	@echo "  make activate  - Show commands to activate the virtual environment"
	@echo "  make clean     - Remove virtual environment"
	@echo ""
	@echo "Activation commands:"
	@echo "  • macOS/Linux: source $(VENV)/bin/activate"
	@echo "  • Windows CMD: $(VENV)\\Scripts\\activate.bat"
	@echo "  • Windows PowerShell: $(VENV)\\Scripts\\Activate.ps1"

# -----------------------------
# Setup environment
# -----------------------------
.PHONY: setup
setup: $(VENV)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Creating .env file..."
	@touch .env
	@if ! grep -q "OPENAI_API_KEY" .env; then \
		echo "OPENAI_API_KEY=TODO" >> .env; \
	fi
	@echo "✅ Environment setup complete!"

# -----------------------------
# Create virtual environment
# -----------------------------
$(VENV):
	@$(PY) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || \
		{ echo "❌ Python 3.10+ is required, found: $$($(PY) --version 2>&1)"; \
		  echo "   Install a newer Python, then run: make setup PY=python3.12"; exit 1; }
	@echo "Creating virtual environment..."
	$(PY) -m venv $(VENV)

# -----------------------------
# Clean virtual environment
# -----------------------------
.PHONY: clean
clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Done."

# -----------------------------
# Show activation command based on system
# -----------------------------
.PHONY: activate
activate:
	@echo "Use the appropriate command to activate your virtual environment:"
	@if [ "$$(uname)" = "Darwin" ] || [ "$$(uname)" = "Linux" ]; then \
		echo "  source $(VENV)/bin/activate"; \
	elif [ "$$(OS)" = "Windows_NT" ]; then \
		echo "  CMD: $(VENV)\Scripts\activate.bat"; \
		echo "  PowerShell: $(VENV)\Scripts\Activate.ps1"; \
	else \
		echo "  Unknown OS, please activate manually"; \
	fi
