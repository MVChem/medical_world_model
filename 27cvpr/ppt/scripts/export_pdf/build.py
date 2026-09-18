"""Export a tracked PowerPoint figure to a local PDF with LibreOffice."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


def export(source: Path, output: Path | None = None) -> Path:
    source = source.resolve(strict=True)
    output = output.resolve() if output else source.with_suffix('.pdf')
    executable = shutil.which('libreoffice') or shutil.which('soffice')
    if not executable:
        raise RuntimeError('LibreOffice is required to export the PowerPoint figures.')
    with TemporaryDirectory(prefix='medworld_pdf_export_') as directory:
        work = Path(directory)
        process = subprocess.run(
            [executable, '-env:UserInstallation=' + (work / 'profile').as_uri(),
             '--headless', '--convert-to', 'pdf', '--outdir', str(work), str(source)],
            capture_output=True, text=True, timeout=120,
        )
        generated = work / source.with_suffix('.pdf').name
        if process.returncode or not generated.is_file():
            raise RuntimeError(process.stdout + process.stderr)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated, output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(export(args.source, args.output))
