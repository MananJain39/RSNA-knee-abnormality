.PHONY: setup install test lint format extract-labels validate clean

PYTHON = /home/manan/.venv/bin/python3
PIP = $(PYTHON) -m pip

setup:
	$(PIP) install -e ".[dev]"

install:
	$(PIP) install pandas numpy scikit-learn pydicom openai pyyaml tqdm pytest

test:
	cd /home/manan/dev/rsna-knee && $(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ scripts/ tests/

format:
	$(PYTHON) -m black src/ scripts/ tests/

extract-labels-regex:
	$(PYTHON) scripts/extract_labels.py --method regex

extract-labels-llm:
	$(PYTHON) scripts/extract_labels.py --method llm --model gpt-4o

validate:
	$(PYTHON) scripts/validate_labels.py

profile-data:
	$(PYTHON) -c "from src.data.loader import load_train, load_train_series; \
		t = load_train(); s = load_train_series(); \
		print(f'Train: {t.shape}'); print(f'Series: {s.shape}'); \
		print(f'Gold studies: {t[t.ACL.notna()].shape[0]}'); \
		print(f'Reports: {t.Report.notna().sum()}')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
