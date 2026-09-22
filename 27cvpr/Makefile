LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error
PYTHON ?= python3
MAIN_FIGURES := ppt/ppt/fig1_v1.pdf ppt/ppt/fig1_v8.pdf ppt/ppt/fig3_v3.pdf \
                ppt/source/fig4_v1.png ppt/source/fig5_v2.png ppt/source/fig6_v1.png
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

ppt/ppt/%.pdf: ppt/ppt/%.pptx ppt/scripts/export_pdf/build.py
	$(PYTHON) ppt/scripts/export_pdf/build.py $<

$(APPENDIX_FIGURE): ppt/scripts/appendix_fig1_v1/build.py $(wildcard ppt/scripts/appendix_fig1_v1/assets/*.jpg)
	$(PYTHON) ppt/scripts/appendix_fig1_v1/build.py

clean:
	latexmk -C main.tex supplementary.tex
	rm -f main.bbl main.brf supplementary.bbl supplementary.brf
