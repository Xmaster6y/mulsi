# CI
.PHONY: checks
checks:
	poetry run pre-commit run --all-files

.PHONY: test-assets
test-assets:
	bash assets/resolve-test-assets.sh

.PHONY: tests
tests:
	poetry run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v

# API
.PHONY: app-start
app-start:
	poetry run python -m demo.main
