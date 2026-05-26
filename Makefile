MKDOCS ?= $(shell [ -x ./docs/bin/mkdocs ] && echo ./docs/bin/mkdocs || echo mkdocs)
DEV_ADDR ?= 127.0.0.1:8000

.PHONY: serve serve-fast serve-clean build quality quality-hints quality-pages paper-quality image-quality clean-appledouble

serve: clean-appledouble
	$(MKDOCS) serve --livereload --open --dev-addr=$(DEV_ADDR)

serve-fast: clean-appledouble
	$(MKDOCS) serve --livereload --dirty --open --dev-addr=$(DEV_ADDR)

serve-clean: clean-appledouble
	$(MKDOCS) serve --livereload --open --dev-addr=$(DEV_ADDR)

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

clean-appledouble:
	find files -name '._*' -type f -delete
