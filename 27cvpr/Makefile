LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error
PYTHON ?= python3
INKSCAPE ?= inkscape
MAIN_FIGURES := ppt/ppt/fig1_v1.pdf ppt/ppt/fig1_v8.pdf ppt/ppt/fig3_v3.pdf \
                ppt/ppt/fig4_v1.pdf ppt/source/fig5_v2.png ppt/ppt/fig6_v1.pdf
APPENDIX_FIGURE := ppt/ppt/appendix_fig1_v1.pdf

.PHONY: all paper supplementary figures clean

all: paper supplementary

# latexmk tracks included TeX, bibliography, style, and figure dependencies.
paper: $(MAIN_FIGURES)
	$(LATEXMK) main.tex

# Import main.aux for references to the manuscript's tables.
supplementary: paper $(APPENDIX_FIGURE)
	$(LATEXMK) supplementary.tex

figures: $(MAIN_FIGURES) $(APPENDIX_FIGURE)

# The experiment SVG embeds its four radiographs; rebuilding needs no run/data folder.
ppt/ppt/fig4_v1.pdf: ppt/ppt/fig4_v1.svg
	$(INKSCAPE) $< --export-area-page --export-type=pdf --export-filename=$@

# The retrieval SVG embeds its radiographs and needs no experiment data to export.
ppt/ppt/fig6_v1.pdf: ppt/ppt/fig6_v1.svg
	$(INKSCAPE) $< --export-area-page --export-type=pdf --export-filename=$@

ppt/ppt/%.pdf: ppt/ppt/%.pptx ppt/scripts/export_pdf/build.py
	$(PYTHON) ppt/scripts/export_pdf/build.py $<

$(APPENDIX_FIGURE): ppt/scripts/appendix_fig1_v1/build.py $(wildcard ppt/scripts/appendix_fig1_v1/assets/*.jpg)
	$(PYTHON) ppt/scripts/appendix_fig1_v1/build.py

clean:
	latexmk -C main.tex supplementary.tex
	rm -f main.bbl main.brf supplementary.bbl supplementary.brf
