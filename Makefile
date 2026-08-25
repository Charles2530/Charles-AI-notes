MKDOCS ?= $(shell [ -x ./docs/bin/mkdocs ] && echo ./docs/bin/mkdocs || echo mkdocs)
DEV_ADDR ?= 127.0.0.1:8000
SERVE_LOCAL = python3 scripts/serve_local.py --mkdocs "$(MKDOCS)" --dev-addr "$(DEV_ADDR)"

.PHONY: serve serve-fast serve-clean build quality quality-hints quality-pages paper-quality image-quality blog-quality math-quality clean-appledouble

serve: clean-appledouble
	$(SERVE_LOCAL) --open

serve-fast: clean-appledouble
	$(SERVE_LOCAL) --dirty --open

serve-clean: clean-appledouble
	$(SERVE_LOCAL) --open

build: clean-appledouble
	$(MKDOCS) build --strict

quality:
	python scripts/content_quality_report.py
	python scripts/paper_deep_dive_audit.py

quality-hints:
	python scripts/content_quality_report.py --include-hints
	python scripts/paper_deep_dive_audit.py --include-hints

quality-pages:
	python scripts/content_quality_report.py --include-hints --list-pages
	python scripts/paper_deep_dive_audit.py --include-hints

paper-quality:
	python scripts/paper_deep_dive_audit.py

image-quality:
	python scripts/image_quality_report.py --contact-sheet --fail-on-high

blog-quality:
	python scripts/blog_readiness_report.py --list-pages

math-quality:
	python scripts/math_rendering_report.py --fail-on-issues --list

clean-appledouble:
	find files -name '._*' -type f -delete
