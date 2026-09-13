# appendix_fig_v3

Build the two-row longitudinal chest-radiograph figure as a publication-sized PDF:

```bash
/home/data2/chk/workspace/2026/.venv/bin/python build_appendix_fig_v3.py
```

The script reads the curated radiographs from `../../source/` and `../../prompt/`, then writes:

```text
../../ppt/appendix_fig_v3.pdf
```

Run the command from this directory. Use `--output PATH` to write the PDF elsewhere.
