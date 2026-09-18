"""Serve the production React build and embed it into standalone HTML exports."""

import re

from fastapi import HTTPException

from ..data import ROOT

DIST = ROOT / "frontend" / "dist"


def offline_html(snapshot):
    if not (DIST / "index.html").is_file():
        raise HTTPException(503, "Build the React frontend before exporting HTML")
    html = (DIST / "index.html").read_text()

    def asset_path(url):
        path = (DIST / url.lstrip("/")).resolve()
        if not path.is_relative_to(DIST.resolve()) or not path.is_file():
            raise HTTPException(
                503, "Missing frontend asset; rebuild the React frontend"
            )
        return path

    def style(match):
        return "<style>" + asset_path(match[1]).read_text() + "</style>"

    scripts = []

    def script(match):
        code = asset_path(match[1]).read_text().replace("</script", "<\\/script")
        scripts.append('<script type="module">' + code + "</script>")
        return ""

    html = re.sub(r'<link\b[^>]*rel="stylesheet"[^>]*href="([^"]+)"[^>]*>', style, html)
    html = re.sub(r'<script\b[^>]*src="([^"]+)"[^>]*></script>', script, html)
    # Vite's entry module now executes after the embedded snapshot and DOM exist.
    return html.replace(
        "</body>",
        "<script>window.MIMIC_SNAPSHOT="
        + snapshot
        + ";</script>"
        + "".join(scripts)
        + "</body>",
    )
