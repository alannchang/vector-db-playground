# vector-db-playground

Here's a comprehensive README.md template:

````markdown:README.md
# Vector DB API

A FastAPI-based vector database API for document embeddings and similarity search.

## Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

### First-time Setup

1. Clone the repository
```bash
git clone <repository-url>
cd vector-db-api
````

2. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies

```bash
# Install production dependencies
pip install -e .

# Install development dependencies (includes black, isort, pylint, pytest)
pip install -e ".[dev]"
```

### Development

1. Always activate virtual environment before working

```bash
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

2. Run the API locally

```bash
uvicorn app.main:app --reload
```

3. Code formatting and linting

```bash
# Format code
black .

# Sort imports
isort .

# Run linter
pylint app/
```

### Dependency Management

If you need to add new dependencies:

1. Add them to `pyproject.toml` under `[project.dependencies]` or `[project.optional-dependencies]`
2. Regenerate requirements file:

```bash
pip install pip-tools
pip-compile pyproject.toml
```

3. Install updated dependencies:

```bash
pip install -e ".[dev]"
```

### Common Issues

- If you see import errors, ensure your virtual environment is activated
- If new dependencies aren't found, run `pip install -e ".[dev]"` again
- If linting fails, run `black .` and `isort .` before committing

## API Documentation

Once running, view the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

```

This provides a clear path for new team members to get started and maintain the project. Feel free to customize based on your specific project needs!
```
