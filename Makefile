# -----------------------------
# Makefile for Python project setup
# -----------------------------

# Name of virtual environment folder
VENV := venv

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
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)

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
