echo "=== Running tests ==="
cd ..
pytest --mpl --cov=PyFields --cov-report=html
$BROWSER htmlcov/index.html &
