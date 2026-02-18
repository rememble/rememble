.PHONY: dev test lint fmt check bump publish release

BUMP ?= patch

dev: ## Install in development mode
	uv sync --all-extras
	@echo "✓ Development mode ready"

test: ## Run tests
	uv run pytest -v

lint: ## Lint + type check
	uv run ruff check src/ tests/
	uv run basedpyright src/

fmt: ## Format + fix imports
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

check: fmt lint test ## Format, lint, type check, test

bump: ## Bump version (BUMP=major|minor|patch)
	uv version --bump $(BUMP)

publish: ## Build + publish to PyPI
	rm -rf dist/
	uv build
	uv publish

release: check ## Full release: fmt, lint, test, bump, tag, push, publish
	@if [ -n "$$(git status --porcelain)" ]; then echo "ERROR: dirty working tree" && exit 1; fi
	uv version --bump $(BUMP)
	$(eval VERSION := $(shell uv version --short))
	git add pyproject.toml uv.lock
	git commit -m "chore(release): v$(VERSION)"
	git tag "v$(VERSION)"
	git push && git push --tags
	$(MAKE) publish

.DEFAULT_GOAL := dev
