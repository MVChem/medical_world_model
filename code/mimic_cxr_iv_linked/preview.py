"""Standalone preview of original linked cases; full originals remain untouched."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from common import read_jsonl


def main(args):
    cases=list(read_jsonl(args.out/'review_cases.jsonl'))
    chosen=[cases[i] for i in [0,1,4] if i<len(cases)]
    fig,axes=plt.subplots(len(chosen),2,figsize=(10,4.3*len(chosen)),squeeze=False)
    for row,c in enumerate(chosen):
        p=c['pair'];context=p.get('current_clinical_availability')
        counts=context['history_counts'] if context else {}
        for col,side in enumerate(['source','target']):
            ax=axes[row,col]
            with Image.open(c[side]['image']['path']) as im:
                ax.imshow(im,cmap='gray')
            ax.axis('off')
            ax.set_title(('Current' if side=='source' else f'Observed follow-up (+{p["realized_gap_hours"]:.1f} h)')+
                f' | {c[side]["image"]["view"]}',fontsize=11)
        axes[row,0].text(0,-.035,f'Case {row+1} | same admission {p["hadm_id"]}',transform=axes[row,0].transAxes,fontsize=9)
        axes[row,1].text(0,-.035,f'Before current: labs={counts.get("labs",0)}, ICU chart={counts.get("icu_chart",0)}',transform=axes[row,1].transAxes,fontsize=9)
    fig.suptitle('MIMIC-CXR + MIMIC-IV: linked reference examples',fontsize=15)
    fig.text(.5,.008,'Original image previews. Full reports, quality flags and timestamp-separated clinical records: review.html',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.025,1,.975])
    fig.savefig(args.out/'review_preview.png',dpi=150,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
