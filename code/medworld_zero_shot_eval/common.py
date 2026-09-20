import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))


def request(endpoint, payload=None, route="/v1/chat/completions", timeout=180):
    req = urllib.request.Request(
        endpoint + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return json.load(f)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:1000]}") from e


def verify(run):
    protocol = read(run / "protocol.json")
    for filename, expected in protocol["file_sha256"].items():
        if digest(run / filename) != expected:
            raise ValueError(f"Frozen artifact changed: {filename}")
    for filename, expected in read(run / "source_manifest.json").items():
        if digest(run / "source" / filename) != expected:
            raise ValueError(f"Frozen executable changed: {filename}")
    return protocol


def prompt(question, vocabulary):
    return (
        "Answer the question using only this chest radiograph. Return only a JSON array of "
        'answer labels from the fixed vocabulary below. For yes/no questions return ["yes"] '
        'or ["no"]. For other questions return all applicable answer labels; use [] if none '
        "apply. Use exact vocabulary labels, without explanations.\n"
        "Vocabulary: " + json.dumps(vocabulary) + "\nQuestion: " + question
    )


def answer_schema(vocabulary):
    return {
        "type": "array",
        "items": {"type": "string", "enum": list(vocabulary)},
        "minItems": 0,
        "maxItems": len(vocabulary),
    }


def inference_prompt(question, vocabulary, style=None):
    if style == "question_only":
        return question
    if style == "llava_short_answer":
        return question + "\nAnswer the question using a single word or phrase."
    return prompt(question, vocabulary)
