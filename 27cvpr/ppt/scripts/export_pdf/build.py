"""Export an editable PPTX to an adjacent local PDF via LibreOffice."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def export_pdf(source: Path) -> Path:
    source = source.resolve(strict=True)
    office = shutil.which('libreoffice') or shutil.which('soffice')
    if not office:
        raise RuntimeError('LibreOffice is required to export PowerPoint figures.')
    if source.suffix.lower() != '.pptx':
        raise ValueError('Expected a .pptx source file.')
    target = source.with_suffix('.pdf')
    with tempfile.TemporaryDirectory(prefix='paper_figure_export_') as directory:
        work = Path(directory)
        subprocess.run([
            office, '-env:UserInstallation=' + (work / 'profile').as_uri(),
            '--headless', '--convert-to', 'pdf', '--outdir', str(work), str(source),
        ], check=True, timeout=90)
        result = work / target.name
        if not result.is_file() or result.stat().st_size == 0:
            raise RuntimeError(f'LibreOffice did not create {target.name}.')
        shutil.copyfile(result, target)
    print(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    export_pdf(args.source)
