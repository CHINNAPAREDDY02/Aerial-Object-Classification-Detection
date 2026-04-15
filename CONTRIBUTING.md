# Contributing Guide

## Project Structure

Follow the existing module structure when adding new code:

- `src/` — Core Python modules (models, data, utilities)
- `scripts/` — Standalone training & utility scripts
- `app/` — Streamlit web application
- `notebooks/` — Jupyter exploration notebooks
- `tests/` — pytest unit tests
- `yolov8/` — Object detection pipeline

## Adding a New Model

1. Add the builder function in `src/transfer_learning.py`
2. Add its path constant in `src/config.py`
3. Update `scripts/train_all.py` to include the new model
4. Add an evaluation call in `notebooks/05_Model_Evaluation_and_Comparison.ipynb`

## Code Style

- Follow PEP 8, max line length 110
- Docstrings on all public functions (NumPy style)
- Type hints encouraged

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Running the Full Pipeline

```bash
make setup    # install + generate labels
make train    # train all models
make app      # launch Streamlit
```
