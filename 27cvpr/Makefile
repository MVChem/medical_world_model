LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all paper supplementary clean plan plan-preview

all: paper supplementary

# latexmk tracks included TeX, bibliography, style, and figure dependencies.
paper:
	$(LATEXMK) main.tex

# Import main.aux for references to the manuscript's tables.
supplementary: paper
	$(LATEXMK) supplementary.tex

plan:
	$(LATEXMK) -outdir=plans plans/table1_table2_plan.tex

plan-preview: plan
	python plans/render_table_previews.py

clean:
	latexmk -C main.tex supplementary.tex
	latexmk -C -outdir=plans plans/table1_table2_plan.tex
