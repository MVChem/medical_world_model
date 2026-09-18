"""Stream patient-level MIMIC-IV tables into memory and query observed records.

No table is assumed to be ordered by subject. CSV parsing preserves quoted
commas/newlines. Original values, units, identifiers and timestamps are retained.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import logging
import math
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .memory import DiscardedLoad, deep_size

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableSpec:
    name: str
    module: str
    title: str
    group: str
    times: tuple[str, ...] = ("charttime", "chartdate")
    end: str = ""
    labels: tuple[str, ...] = ()
    value: str = "value"
    number: str = "valuenum"
    unit: str = "valueuom"
    dictionary: str = ""
    parent: str = ""
    parent_key: str = ""
    note: str = ""


SPECS = [
    TableSpec(
        "labevents",
        "hosp",
        "检验",
        "measurements",
        dictionary="d_labitems",
        note="原始检验结果与参考区间；保留缺失 hadm_id 的记录，不推断住院。",
    ),
    TableSpec(
        "chartevents",
        "icu",
        "生命体征 / ICU 记录",
        "measurements",
        dictionary="d_items",
        note="含生命体征、呼吸机参数、评分和其他 ICU 记录；不与检验表自动合并。",
    ),
    TableSpec(
        "outputevents",
        "icu",
        "排出量",
        "measurements",
        number="value",
        dictionary="d_items",
        note="保留原始记录值，不计算累计出量或液体平衡。",
    ),
    TableSpec(
        "microbiologyevents",
        "hosp",
        "微生物 / 药敏",
        "laboratory",
        labels=("test_name", "spec_type_desc"),
        value="org_name",
        number="",
        note="保留标本、菌种、抗生素与药敏解释；未长菌、未记录和未完成不互相替代。",
    ),
    TableSpec(
        "omr",
        "hosp",
        "门诊 / 常规测量",
        "measurements",
        labels=("result_name",),
        value="result_value",
        number="result_value",
        unit="",
        note="chartdate 只有日期精度；复合血压值保留原文，不拆成推测数值。",
    ),
    TableSpec(
        "prescriptions",
        "hosp",
        "处方",
        "medications",
        times=("starttime",),
        end="stoptime",
        labels=("drug",),
        value="dose_val_rx",
        number="",
        unit="dose_unit_rx",
        note="处方剂量与计划起止时间，不代表实际给药。",
    ),
    TableSpec(
        "emar",
        "hosp",
        "给药记录",
        "medications",
        labels=("medication",),
        value="event_txt",
        number="",
        note="保留给药、未给药等事件状态；系统覆盖并非所有年代。",
    ),
    TableSpec(
        "pharmacy",
        "hosp",
        "药房医嘱",
        "medications",
        times=("starttime",),
        end="stoptime",
        labels=("medication",),
        value="status",
        number="",
        note="药房订单状态，不当作实际给药。",
    ),
    TableSpec(
        "poe",
        "hosp",
        "医嘱",
        "orders",
        times=("ordertime",),
        labels=("order_type",),
        value="order_subtype",
        number="",
        note="含新建、修改和停止医嘱；原始状态完整保留。",
    ),
    TableSpec(
        "emar_detail",
        "hosp",
        "给药明细",
        "medications",
        times=(),
        labels=("product_description", "administration_type"),
        value="dose_given",
        number="",
        unit="dose_given_unit",
        parent="emar",
        parent_key="emar_id",
        note="通过 subject_id + emar_id 关联给药时刻和住院；保留分次产品行，不相加或去重。",
    ),
    TableSpec(
        "poe_detail",
        "hosp",
        "医嘱明细",
        "orders",
        times=(),
        labels=("field_name",),
        value="field_value",
        number="",
        parent="poe",
        parent_key="poe_id",
        note="通过 subject_id + poe_id 关联订单时间和住院；父记录缺失或歧义会单独保留。",
    ),
    TableSpec(
        "datetimeevents",
        "icu",
        "ICU 日期型记录",
        "icu",
        dictionary="d_items",
        number="",
        note="charttime 为记录时刻，value 为记录中的日期值；两者分别保留。",
    ),
    TableSpec(
        "ingredientevents",
        "icu",
        "输入成分",
        "icu",
        times=("starttime",),
        end="endtime",
        value="amount",
        number="amount",
        unit="amountuom",
        dictionary="d_items",
        note="输入成分量与速率；不与 inputevents 求和，避免重复计量。",
    ),
    TableSpec(
        "services",
        "hosp",
        "临床服务转移",
        "administrative",
        times=("transfertime",),
        labels=("curr_service",),
        value="prev_service",
        number="",
    ),
    TableSpec(
        "hcpcsevents",
        "hosp",
        "HCPCS 操作",
        "administrative",
        labels=("short_description", "hcpcs_cd"),
        value="hcpcs_cd",
        number="",
        dictionary="d_hcpcs",
    ),
    TableSpec(
        "drgcodes",
        "hosp",
        "DRG 编码",
        "administrative",
        times=(),
        labels=("description",),
        value="drg_code",
        number="",
        note="住院级编码，无独立事件时间。",
    ),
    TableSpec(
        "patients",
        "hosp",
        "患者信息",
        "administrative",
        times=(),
        labels=(),
        value="gender",
        number="",
        note="anchor_age 是锚定年份年龄；本界面不据此推算某次检查年龄。",
    ),
]
TABLES = {spec.name: spec for spec in SPECS}


def table_path(root: Path, spec: TableSpec) -> Path:
    path = root / spec.module / f"{spec.name}.csv.gz"
    return path if path.is_file() else path.with_suffix("")


def header(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream))


def scan_subjects(
    path, subjects, progress=None, stop=None, block_size=16 * 1024 * 1024
):
    """Bounded Arrow batches; scan the whole source, never rely on subject ordering."""
    fields = header(path)
    if "subject_id" not in fields:
        raise ValueError(f"No subject_id in {path.name}")
    selected = defaultdict(list)
    scanned = 0
    subjects_array = pa.array(sorted(subjects), type=pa.string())
    with pa.OSFile(str(path), "rb") as raw:
        source = pa.CompressedInputStream(raw, "gzip") if path.suffix == ".gz" else raw
        reader = pacsv.open_csv(
            source,
            read_options=pacsv.ReadOptions(block_size=block_size, use_threads=False),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                column_types={name: pa.string() for name in fields}
            ),
        )
        try:
            for batch in reader:
                if stop is not None and stop.is_set():
                    raise InterruptedError("Data scan stopped")
                scanned += batch.num_rows
                subset = batch.filter(
                    pc.is_in(batch.column("subject_id"), value_set=subjects_array)
                )
                for row in subset.to_pylist():
                    selected[row["subject_id"]].append(row)
                if progress:
                    progress(scanned, raw.tell())
        finally:
            reader.close()
    return selected, fields, scanned


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def timestamp(value):
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value))
        # MIMIC timestamps are local, de-identified and timezone-naive.
        return result if result.tzinfo is None else None
    except ValueError:
        return None


def normalize(spec, rows, dictionary, parents=()):
    parent_index = defaultdict(list)
    for row in parents:
        parent_index[(row.get("subject_id"), row.get(spec.parent_key))].append(row)
    events = []
    for index, row in enumerate(rows):
        item_id = row.get("itemid") or row.get("hcpcs_cd") or ""
        info = dictionary.get(item_id, {})
        label = (
            info.get("label")
            or info.get("long_description")
            or next((row.get(k) for k in spec.labels if row.get(k)), None)
            or spec.title
        )
        time_field = next((key for key in spec.times if row.get(key)), "")
        at = row.get(time_field, "")
        hadm = row.get("hadm_id", "")
        parent_link = "not_applicable"
        storetime = row.get("storetime", "")
        if spec.parent:
            matches = parent_index.get(
                (row.get("subject_id"), row.get(spec.parent_key)), []
            )
            parent_link = (
                "unique"
                if len(matches) == 1
                else "ambiguous"
                if matches
                else "unmatched"
            )
            if len(matches) == 1:
                parent = matches[0]
                time_field = "charttime" if spec.parent == "emar" else "ordertime"
                at, hadm = parent.get(time_field, ""), parent.get("hadm_id", "")
                storetime = parent.get("storetime", "")
                time_field = spec.parent + "." + time_field
        raw_value = row.get(spec.value, "")
        value = number(row.get(spec.number)) if spec.number else None
        if any(op in str(raw_value) for op in ("<", ">")):
            value = None  # A bound is not an exact point measurement.
        unit = row.get(spec.unit, "") if spec.unit else ""
        series_id = hashlib.sha256(
            json_key(item_id or label, unit).encode()
        ).hexdigest()[:16]
        valid_time = timestamp(at)
        end_value = row.get(spec.end, "") if spec.end else ""
        events.append(
            {
                "row_index": index,
                "label": label,
                "item_id": item_id,
                "category": info.get("category", ""),
                "fluid": info.get("fluid", ""),
                "time": valid_time.isoformat() if valid_time else None,
                "time_precision": "date"
                if valid_time and len(at) == 10
                else "timestamp"
                if valid_time
                else "unknown",
                "time_source": time_field,
                "end_time": timestamp(end_value).isoformat()
                if timestamp(end_value)
                else None,
                "storetime": storetime or None,
                "value": raw_value,
                "numeric_value": value,
                "unit": unit,
                "series_id": series_id,
                "hadm_id": hadm or None,
                "stay_id": row.get("stay_id") or None,
                "flag": row.get("flag") or row.get("warning") or "",
                "reference_lower": row.get("ref_range_lower", ""),
                "reference_upper": row.get("ref_range_upper", ""),
                "parent_link": parent_link,
                "raw": row,
            }
        )
    return sorted(events, key=lambda row: (row["time"] or "9999", row["row_index"]))


def json_key(item, unit):
    # Length prefixes prevent collisions when a label itself contains separators.
    return f"{len(item)}:{item}{len(unit)}:{unit}"


def filter_events(
    events, *, scope="patient", start=None, end=None, hadm_id="", q="", series_id=""
):
    if scope not in {"patient", "window", "admission"}:
        raise ValueError("Unknown scope")
    lower, upper = timestamp(start), timestamp(end)
    if scope == "window" and (lower is None or upper is None or lower > upper):
        raise ValueError("A valid start/end window is required")
    if scope == "admission" and not hadm_id:
        raise ValueError("Choose an admission")
    matched = []
    query = q.casefold().strip()
    for row in events:
        if scope == "admission" and row["hadm_id"] != hadm_id:
            continue
        if scope == "window":
            at = timestamp(row["time"])
            if at is None:
                continue
            until = timestamp(row["end_time"]) or at
            if row["time_precision"] == "date":
                # Day precision overlaps a day; never invent a midnight observation.
                until = at + timedelta(days=1) - timedelta(microseconds=1)
            if at > upper or until < lower:
                continue
        if series_id and row["series_id"] != series_id:
            continue
        if (
            query
            and query
            not in " ".join(
                str(v)
                for v in (
                    row["label"],
                    row["category"],
                    row["item_id"],
                    *row["raw"].values(),
                )
            ).casefold()
        ):
            continue
        matched.append(row)
    return matched


def summarize(events):
    series = {}
    for row in events:
        if row["numeric_value"] is None or row["time_precision"] != "timestamp":
            continue
        key = row["series_id"]
        if key not in series:
            series[key] = {
                "id": key,
                "label": row["label"],
                "item_id": row["item_id"],
                "unit": row["unit"],
                "category": row["category"],
                "count": 0,
            }
        series[key]["count"] += 1
    # Vital-sign dictionary category first; otherwise deterministic label order.
    return sorted(
        series.values(),
        key=lambda s: (
            "vital" not in s["category"].lower(),
            s["label"],
            s["item_id"],
            s["unit"],
        ),
    )


def chart_points(events, series_id, limit=1500):
    points = [
        r
        for r in events
        if r["series_id"] == series_id
        and r["numeric_value"] is not None
        and r["time_precision"] == "timestamp"
    ]
    total = len(points)
    if total > limit:
        # Retain extrema per chronological bucket, endpoints and announce sampling.
        step = math.ceil(total / ((limit - 2) // 2))
        selected = {0, total - 1}
        for i in range(0, total, step):
            indexes = range(i, min(i + step, total))
            selected.add(min(indexes, key=lambda j: points[j]["numeric_value"]))
            selected.add(max(indexes, key=lambda j: points[j]["numeric_value"]))
        points = [points[i] for i in sorted(selected)]
    keys = (
        "time",
        "time_precision",
        "numeric_value",
        "value",
        "unit",
        "hadm_id",
        "stay_id",
        "flag",
        "reference_lower",
        "reference_upper",
        "storetime",
    )
    return {
        "points": [{k: r[k] for k in keys} for r in points],
        "total": total,
        "sampled": len(points) < total,
    }


class ClinicalTables:
    """Two bounded scan workers with per-patient/per-table deduplication."""

    def __init__(self, root, index=None, memory=None):
        self.root = Path(root)
        self.index = index
        self.memory = memory
        self.lock = memory.lock if memory else threading.RLock()
        self.stop = threading.Event()
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="iv-large-table"
        )
        self.states, self.rows, self.events, self.fields, self.futures = (
            {},
            {},
            {},
            {},
            {},
        )
        self.dictionaries = {}

    def close(self):
        self.stop.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self.lock:
            for mapping in (
                self.states,
                self.rows,
                self.events,
                self.fields,
                self.futures,
                self.dictionaries,
            ):
                mapping.clear()

    def evict(self, subject):
        with self.lock:
            for mapping in (self.states, self.rows, self.events, self.fields):
                for key in [k for k in mapping if k[0] == subject]:
                    mapping.pop(key, None)
            for key in [k for k in self.futures if k[0] == subject]:
                future = self.futures.pop(key, None)
                if future:
                    future.cancel()

    def _current(self, subject, token):
        return not self.stop.is_set() and (
            self.memory is None or self.memory.current(subject, token)
        )

    def manifest(self, subject):
        counts = self.index.directory(subject) if self.index else {}
        with self.lock:
            output = []
            for spec in SPECS:
                path = table_path(self.root, spec)
                indexed = (
                    self.index is not None
                    and spec.module + "." + spec.name in self.index.tables
                )
                available = indexed if self.index else path.is_file()
                status = self.states.get(
                    (subject, spec.name),
                    {"state": "not_loaded" if available else "unavailable"},
                )
                output.append(
                    {
                        "name": spec.name,
                        "title": spec.title,
                        "module": spec.module,
                        "group": spec.group,
                        "note": spec.note,
                        "parent": spec.parent,
                        "bytes": path.stat().st_size if path.is_file() else 0,
                        **(
                            {
                                "count": counts.get(spec.module + "." + spec.name, 0),
                                "indexed": True,
                                "read_method": "patient_index",
                            }
                            if indexed
                            else {}
                        ),
                        **status,
                    }
                )
            return output

    def request(self, subjects, names):
        if any(name not in TABLES for name in names):
            raise ValueError("Unknown clinical table")
        with self.lock:
            for name in names:
                spec = TABLES[name]
                if spec.parent:
                    self.request(subjects, [spec.parent])
                pending = {
                    s
                    for s in subjects
                    if self.states.get((s, name), {}).get("state")
                    not in {"ready", "queued", "loading"}
                }
                if not pending:
                    continue
                path = table_path(self.root, spec)
                if not path.is_file():
                    for subject in pending:
                        self.states[subject, name] = {
                            "state": "unavailable",
                            "error": "源表不存在",
                        }
                    continue
                for subject in pending:
                    self.states[subject, name] = {
                        "state": "queued",
                        "percent": 0,
                        "rows_scanned": 0,
                    }
                tokens = {
                    s: self.memory.entries[s].token if self.memory else None
                    for s in pending
                }
                future = self.executor.submit(self._load, spec, tokens, path)
                keys = [(subject, name) for subject in pending]
                for key in keys:
                    self.futures[key] = future

                def done(completed, keys=keys):
                    with self.lock:
                        for key in keys:
                            if self.futures.get(key) is completed:
                                self.futures.pop(key, None)

                future.add_done_callback(done)

    def _dictionary(self, spec):
        if not spec.dictionary:
            return {}
        with self.lock:
            if spec.dictionary in self.dictionaries:
                return self.dictionaries[spec.dictionary]
        module = "icu" if spec.dictionary == "d_items" else "hosp"
        path = table_path(self.root, TableSpec(spec.dictionary, module, "", ""))
        if self.index:
            values = {
                r.get("itemid") or r.get("code"): r
                for r in self.index.iter_table(module + "." + spec.dictionary)
            }
        else:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
                values = {
                    r.get("itemid") or r.get("code"): r for r in csv.DictReader(stream)
                }
        with self.lock:
            self.dictionaries[spec.dictionary] = values
        return values

    def _load(self, spec, tokens, path):
        name, size = spec.name, path.stat().st_size

        def live():
            return {s for s, t in tokens.items() if self._current(s, t)}

        def progress(scanned, position):
            with self.lock:
                subjects = live()
                if not subjects:
                    raise DiscardedLoad()
                for subject in subjects:
                    self.states[subject, name] = {
                        "state": "loading",
                        "rows_scanned": scanned,
                        "percent": min(99, round(position / max(1, size) * 100, 1)),
                    }

        try:
            subjects = live()
            if not subjects:
                return
            if spec.parent:
                for subject in subjects:
                    with self.lock:
                        parent_future = self.futures.get((subject, spec.parent))
                    if parent_future:
                        parent_future.result()
                    with self.lock:
                        if not self._current(subject, tokens[subject]):
                            continue
                        if (
                            self.states.get((subject, spec.parent), {}).get("state")
                            != "ready"
                        ):
                            raise ValueError(f"先完成父表 {spec.parent} 的加载")
            progress(0, 0)
            dictionary = self._dictionary(spec)
            if self.index:
                table_name = spec.module + "." + spec.name
                selected = {s: self.index.read_subject(table_name, s) for s in live()}
                fields, scanned = self.index.table(table_name)["fields"], 0
            else:
                selected, fields, scanned = scan_subjects(
                    path, live(), progress, self.stop
                )
            for subject in tokens:
                rows = selected.get(subject, [])
                with self.lock:
                    if not self._current(subject, tokens[subject]):
                        continue
                    parents = self.rows.get((subject, spec.parent), [])
                events = normalize(spec, rows, dictionary, parents)
                estimate = deep_size((rows, events, fields)) if self.memory else 0
                with self.lock:
                    if not self._current(subject, tokens[subject]):
                        continue
                    self.rows[subject, name], self.events[subject, name] = rows, events
                    self.fields[subject, name] = fields
                    self.states[subject, name] = {
                        "state": "ready",
                        "percent": 100,
                        "rows_scanned": scanned,
                        "count": len(rows),
                        "undated": sum(not e["time"] for e in events),
                        "missing_hadm": sum(not e["hadm_id"] for e in events),
                        "read_method": "patient_index" if self.index else "source_scan",
                    }
                    if self.memory:
                        self.memory.account(subject, tokens[subject], name, estimate)
            LOG.info(
                "Loaded %s: retained %s rows for %s subjects",
                name,
                sum(len(rows) for rows in selected.values()),
                len(selected),
            )
        except DiscardedLoad:
            pass
        except Exception as exc:
            with self.lock:
                subjects = live()
                if subjects:
                    LOG.exception("Clinical table read failed: %s", name)
                for subject in subjects:
                    self.states[subject, name] = {"state": "error", "error": str(exc)}

    def query(self, subject, name, *, page=1, limit=50, series_id="", **filters):
        if name not in TABLES:
            raise ValueError("Unknown clinical table")
        with self.lock:
            status = dict(self.states.get((subject, name), {"state": "not_loaded"}))
            events = self.events.get((subject, name), [])
            fields = self.fields.get((subject, name), [])
        filtered = filter_events(events, **filters)
        series = summarize(filtered)
        selected_id = series_id or (series[0]["id"] if series else "")
        shown = (
            [r for r in filtered if r["series_id"] == series_id]
            if series_id
            else filtered
        )
        return {
            "table": name,
            "status": status,
            "total": len(shown),
            "filtered_total": len(filtered),
            "page": page,
            "limit": limit,
            "rows": shown[(page - 1) * limit : page * limit],
            "fields": fields,
            "series": series,
            "selected_series": selected_id,
            "chart": chart_points(filtered, selected_id),
        }

    def export_window(self, subject, start, end):
        lower = (timestamp(start) - timedelta(hours=24)).isoformat()
        upper = (timestamp(end) + timedelta(hours=24)).isoformat()
        output = {
            "start": lower,
            "end": upper,
            "tables": {},
            "manifest": self.manifest(subject),
        }
        with self.lock:
            for spec in SPECS:
                key = (subject, spec.name)
                if self.states.get(key, {}).get("state") != "ready":
                    continue
                events = filter_events(
                    self.events[key], scope="window", start=lower, end=upper
                )
                output["tables"][spec.name] = {
                    "events": events,
                    "fields": self.fields[key],
                    "undated_excluded": self.states[key]["undated"],
                }
        return output
