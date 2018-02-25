echo "=== Running tests ==="
cd ..
python setup.py pytest
pytest --cov=PyFields --cov-report=html
