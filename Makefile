MKDOCS ?= $(shell [ -x ./docs/bin/mkdocs ] && echo ./docs/bin/mkdocs || echo mkdocs)
DEV_ADDR ?= 127.0.0.1:8000

.PHONY: serve serve-fast serve-clean build quality clean-appledouble

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

clean-appledouble:
	find files -name '._*' -type f -delete
