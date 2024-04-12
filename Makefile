# CI
.PHONY: checks
checks:
	poetry run pre-commit run --all-files

.PHONY: test-assets
test-assets:
	bash assets/resolve-test-assets.sh

.PHONY: tests
tests:
	poetry run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=1 -s -v

# API
.PHONY: demo-explore-label-concepts
demo-explore-label-concepts:
	poetry run python explore-label-concepts/app.py
