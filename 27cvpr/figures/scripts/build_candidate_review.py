#!/usr/bin/env python3
"""Combine the four downstream candidate figures without changing the manuscript.

Preserves the native vector PDF panels; overview is a lightweight contact sheet.
"""
from pathlib import Path
import json
import hashlib

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '27cvpr/figures/generated'
FIGURES = [
    ('clinical_evidence', '1. Clinical evidence / baseline attribution'),
    ('state_retrieval', '2. Retrieval / frozen visual depths'),
    ('slot_perturbation', '3. Slot perturbation / fixed decoder'),
    ('spatial_outputs', '4. Spatial outputs / matched frozen controls'),
]


def main():
    document = fitz.open()
    fig, axes = plt.subplots(2, 2, figsize=(18, 13.5), facecolor='#eef1f4')
    manifest = dict(scope='Four downstream candidate figures; empirical first versions with limitations shown in each figure',
                    paper_placeholders='Full-model comparisons remain pending; this review is separate from main.pdf',
                    figures=[])
    for ax, (name, title) in zip(axes.flat, FIGURES):
        source = OUT / f'{name}.pdf'
        with fitz.open(source) as panel:
            assert len(panel) == 1, f'{name} must be one page'
            assert panel[0].get_text().strip(), f'{name} has no searchable text'
            document.insert_pdf(panel)
            pix = panel[0].get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
            pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height,pix.width,pix.n)
            ax.imshow(pixels)
        ax.axis('off');ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=12, color='#172d45')
        manifest['figures'].append(dict(name=name, pdf=str(source.relative_to(ROOT)),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(), page=len(document)))
    document.set_metadata(dict(title='MedWorld-JEPA: four candidate visualizations, first empirical versions',
                               subject='Frozen visual controls and baseline attribution; full proposed method pending'))
    document.save(OUT / 'candidate_figures_v1.pdf', garbage=4, deflate=True)
    document.close()
    fig.subplots_adjust(left=.015,right=.985,top=.94,bottom=.02,hspace=.13,wspace=.035)
    fig.suptitle('First candidate figures — existing data and final checkpoints', fontsize=20, x=.02,ha='left',color='#172d45')
    fig.savefig(OUT / 'candidate_figures_v1_overview.png',dpi=145,facecolor=fig.get_facecolor())
    plt.close(fig)
    (OUT / 'candidate_figures_v1_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(OUT / 'candidate_figures_v1.pdf')


if __name__ == '__main__':
    main()
