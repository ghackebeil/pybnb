#
# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -v
SPHINXBUILD   = sphinx-build
BUILDDIR      = build

ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(SPHINXOPTS) .

.PHONY: help clean html spelling

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  spelling   to check spelling of documentation"

clean:
	-rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

spelling:
	$(SPHINXBUILD) -W -b spelling $(ALLSPHINXOPTS) $(BUILDDIR)/spelling
	@echo
	@echo "Build finished. The spelling pages are in $(BUILDDIR)/spelling."
