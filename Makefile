# AspectFL Makefile
# Build automation and common tasks

.PHONY: help install install-dev test lint format clean docs experiments paper all

# Default target
help:
	@echo "AspectFL - Aspect-Oriented Programming for Federated Learning"
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install AspectFL and dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run test suite"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black"
	@echo "  clean        Clean build artifacts"
	@echo "  docs         Build documentation"
	@echo "  experiments  Run comprehensive experiments"
	@echo "  paper        Build research paper"
	@echo "  all          Run all tasks (install, test, lint, docs)"

# Installation targets
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,experiments]"

# Testing and quality assurance
test:
	python -m pytest tests/ -v --cov=aspectfl --cov-report=html --cov-report=term

lint:
	flake8 aspectfl/ experiments/ tests/
	mypy aspectfl/

format:
	black aspectfl/ experiments/ tests/
	isort aspectfl/ experiments/ tests/

# Documentation
docs:
	cd docs && make html

# Experiments and analysis
experiments:
	python experiments/comprehensive_experiments.py
	python experiments/data_analysis.py

# Research paper
paper:
	cd paper && pdflatex aspectfl_paper.tex && pdflatex aspectfl_paper.tex

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Package building
build:
	python setup.py sdist bdist_wheel

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

# Docker targets
docker-build:
	docker build -t aspectfl:latest .

docker-run:
	docker run -it --rm -v $(PWD):/workspace aspectfl:latest

# Jupyter notebook
notebook:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# All-in-one target
all: install-dev test lint docs

# Development workflow
dev-setup: install-dev
	pre-commit install

# Continuous integration
ci: install test lint

# Performance benchmarks
benchmark:
	python experiments/benchmark.py

# Security scan
security:
	bandit -r aspectfl/
	safety check

# Code coverage
coverage:
	python -m pytest tests/ --cov=aspectfl --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Generate requirements
freeze:
	pip freeze > requirements-frozen.txt

# Update dependencies
update:
	pip install --upgrade -r requirements.txt

# Create release
release: clean test lint build
	@echo "Release package created in dist/"

# Help with common development tasks
dev-help:
	@echo "Development workflow:"
	@echo "1. make dev-setup    # Initial setup"
	@echo "2. make test         # Run tests"
	@echo "3. make lint         # Check code quality"
	@echo "4. make format       # Format code"
	@echo "5. make experiments  # Run experiments"
	@echo "6. make docs         # Build documentation"

