"""Render the shared plan tables directly from PDF, excluding surrounding prose."""
from pathlib import Path
import math
import subprocess
import xml.etree.ElementTree as ET


def main():
    paper_dir = Path(__file__).resolve().parents[1]
    plan_dir = paper_dir / "plans"
    pdf = plan_dir / "table1_table2_plan.pdf"
    xml = subprocess.run(
        ["pdftotext", "-bbox-layout", str(pdf), "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    pages = ET.fromstring(xml).findall(".//{*}page")
    if len(pages) != 2:
        raise ValueError(f"Expected two plan pages, found {len(pages)}")
    dpi = 160
    scale = dpi / 72
    for index, marker in enumerate(("Planned scoring interface.", "Data."), 1):
        lines = []
        for line in pages[index - 1].findall(".//{*}line"):
            text = " ".join(word.text or "" for word in line.findall("{*}word"))
            lines.append((text, {key: float(value) for key, value in line.attrib.items()}))
        top = next(box["yMin"] for text, box in lines if text.startswith(f"Table {index}:"))
        following = next(box["yMin"] for text, box in lines if text.startswith(marker))
        boxes = [box for _, box in lines if box["yMin"] >= top and box["yMax"] < following]
        left = math.floor((min(box["xMin"] for box in boxes) - 4) * scale)
        right = math.ceil((max(box["xMax"] for box in boxes) + 4) * scale)
        upper = math.floor((top - 4) * scale)
        lower = math.ceil((max(box["yMax"] for box in boxes) + 4) * scale)
        output = plan_dir / f"table{index}_preview"
        subprocess.run([
            "pdftoppm", "-f", str(index), "-l", str(index), "-singlefile",
            "-r", str(dpi), "-x", str(left), "-y", str(upper),
            "-W", str(right - left), "-H", str(lower - upper),
            "-png", str(pdf), str(output),
        ], check=True)
        print(f"Rendered {output.with_suffix('.png').relative_to(paper_dir)}")


if __name__ == "__main__":
    main()
