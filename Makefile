# tractus monorepo — workspace Makefile

SHELL := /bin/bash
JOBS ?= $(shell nproc 2>/dev/null || echo 4)

.PHONY: help
help:
	@echo "tractus workspace targets:"
	@echo "  make install   editable install of tractus_shared, tractus_controller, tractus_engine_advanced"
	@echo "  make test      run all test suites"
	@echo "  make ci        full CI surface — lint guardrails + pytest slice (matches GitHub Actions)"
	@echo "  make ci-lint   just the lint guardrails (cheap; same as the pre-commit hooks)"
	@echo "  make doctor    environment + alignment check"
	@echo "  make clean     wipe build/"

.PHONY: install
install:
	pip install -e tractus_shared
	pip install -e tractus_controller
	pip install -e tractus_engine_advanced

.PHONY: test
test:
	@cd tractus_shared && python -m pytest -ra -q
	@cd tractus_controller && python -m pytest -ra -q
	@cd tractus_engine_advanced && python -m pytest -ra -q

.PHONY: doctor
doctor:
	@command -v tractus-controller >/dev/null 2>&1 && tractus-controller doctor || \
	  echo "tractus-controller not on PATH; run 'make install' first"

.PHONY: ci-lint
ci-lint:
	@bash scripts/ci/lint_all.sh

.PHONY: ci
ci: ci-lint
	@cd tractus_engine_advanced && python -m pytest -ra -q
	@cd tractus_controller && python -m pytest -ra -q
	@python -m pytest -ra -q tests/ci

.PHONY: clean
clean:
	rm -rf build/
