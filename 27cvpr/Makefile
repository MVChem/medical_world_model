LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all paper supplementary clean

all: paper supplementary

# latexmk tracks included TeX, bibliography, style, and figure dependencies.
paper:
	$(LATEXMK) main.tex

# Import main.aux for references to the manuscript's tables.
supplementary: paper
	$(LATEXMK) supplementary.tex

clean:
	latexmk -C main.tex supplementary.tex
	rm -f main.bbl main.brf supplementary.bbl supplementary.brf
