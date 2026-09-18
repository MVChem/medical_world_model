"""Bounded caches, stale background work, response leases and reload fidelity."""

import gzip
import threading
import time
from dataclasses import replace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mimic_atlas.app import create_app
from mimic_atlas.memory import CacheBusy, ImageMemory, PatientMemory
from mimic_atlas.prepare_index import build_index
from mimic_atlas.tests.test_atlas import await_patient
from mimic_atlas.tests.test_atlas import sources as source_fixture
from mimic_atlas.tests.test_clinical_tables import (
    await_tables,
)
from mimic_atlas.tests.test_clinical_tables import (
    extended_sources as extended_fixture,
)


def test_patient_lru_ttl_budget_and_leases():
    now, removed = [0.0], []
    cache = PatientMemory(
        threading.RLock(),
        lambda s, t: removed.append(s),
        max_patients=2,
        max_bytes=100,
        ttl=10,
        clock=lambda: now[0],
    )
    a = cache.touch("a")
    cache.touch("b")
    cache.touch("a")
    cache.touch("c")
    assert removed == ["b"] and cache.current("a", a)
    with cache.lease("a"):
        cache.touch("c")
        cache.account("c", cache.entries["c"].token, "rows", 90)
        cache.account("a", a, "rows", 40)
        assert set(cache.entries) == {"a", "c"}  # The in-flight LRU is protected.
    assert list(cache.entries) == ["c"]
    assert not cache.current("a", a)
    cache.account("a", a, "rows", 1000)  # Stale completion cannot restore anything.
    now[0] = 11
    cache.sweep()
    assert not cache.entries
    with cache.lease("a"), cache.lease("b"), pytest.raises(CacheBusy):
        cache.touch("c")
    cache.clear()
    huge = cache.touch("large")
    cache.account("large", huge, "rows", 200)
    assert cache.current(
        "large", huge
    )  # A single large patient is complete, never truncated.
    now[0] = 22
    cache.sweep()
    assert not cache.entries


def test_image_exact_byte_limit_ttl_and_eviction():
    now = [0.0]
    cache = ImageMemory(max_bytes=10, max_entries=2, ttl=5, clock=lambda: now[0])
    cache.put("a", b"1234")
    cache.put("b", b"1234")
    assert cache.get("a") == b"1234"
    cache.put("c", b"12345")
    assert cache.get("b") is None and cache.bytes == 9
    cache.put("too-big", b"x" * 11)
    assert cache.get("too-big") is None and cache.bytes == 9
    now[0] = 6
    cache.sweep()
    assert cache.bytes == 0 and not cache.entries


@pytest.fixture
def indexed(tmp_path):
    _, config, labs = extended_fixture.__wrapped__(source_fixture.__wrapped__(tmp_path))
    for path in config.iv_root.rglob("*.csv.gz"):
        with gzip.open(path, "rb") as stream:
            path.with_suffix("").write_bytes(stream.read())
        path.unlink()
    out = tmp_path / "index"
    build_index(config.iv_root, config.cxr_root, out)
    return replace(config, index_root=out, patient_cache_count=1), labs


def test_evicts_all_patient_maps_and_reloads_original_rows(indexed):
    config, labs = indexed
    with TestClient(create_app(config)) as client:
        await_patient(client)
        await_tables(client)
        store = client.app.state.store
        old = store.memory.entries["10000001"].token
        await_patient(client, "10000002")
        for mapping in (store.patients, store.clinical, store.tables, store.background):
            assert "10000001" not in mapping
        for mapping in (
            store.extended.rows,
            store.extended.events,
            store.extended.fields,
            store.extended.states,
            store.extended.futures,
        ):
            assert not any(k[0] == "10000001" for k in mapping)
        assert not store.memory.current("10000001", old)
        await_patient(client)
        await_tables(client)
        result = client.get(
            "/api/patients/10000001/tables/labevents?scope=patient"
        ).json()
        assert [r["raw"] for r in result["rows"]] == labs
        # A manual clock advance models inactivity; the same sweep runs on a timer.
        store.memory.clock = lambda: time.monotonic() + 1000
        store.memory.sweep()
        assert (
            not store.patients
            and not store.extended.events
            and not store.extended.futures
        )
        assert (
            store.shared_dictionaries
        )  # Global dictionaries are shared, not copied per patient.
    assert not store.shared_dictionaries and not store.memory.entries


def test_idle_timer_releases_records_and_previews_without_requests(indexed):
    config, _ = indexed
    config = replace(config, cache_idle_seconds=1)
    with TestClient(create_app(config)) as client:
        await_patient(client)
        await_tables(client)
        store = client.app.state.store
        store.image_memory.put("idle-preview", b"encoded-image")
        assert store.patients and store.image_memory.entries
        # Do not issue requests or call sweep: the janitor must work on its own.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with store.lock, store.image_memory.lock:
                empty = not store.memory.entries and not store.image_memory.entries
            if empty:
                break
            time.sleep(0.05)
        assert empty
        with store.lock:
            assert not store.patients and not store.clinical and not store.tables
            assert not store.extended.rows and not store.extended.events
    assert not store.janitor.is_alive()


@pytest.mark.parametrize("delayed_table", ["hosp.labevents", "hosp.admissions"])
def test_evicted_running_load_cannot_overwrite_reopened_patient(indexed, delayed_table):
    config, _ = indexed
    with TestClient(create_app(config)) as client:
        # Wait for catalog without allocating a patient.
        for _ in range(300):
            if (
                client.get("/api/catalog").json().get("cohort", {}).get("state")
                == "ready"
            ):
                break
            time.sleep(0.01)
        store = client.app.state.store
        started, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        original = store.index.read_subject
        calls = [0]

        def delayed(name, subject):
            result = original(name, subject)
            if name == delayed_table and subject == "10000001":
                calls[0] += 1
                if calls[0] == 1:
                    started.set()
                    assert release.wait(10)
                    finished.set()
                    return [
                        {**row, "value": "STALE", "hadm_id": "99999999"}
                        for row in result
                    ]
            return result

        with patch.object(store.index, "read_subject", side_effect=delayed):
            try:
                client.get("/api/patients/10000001?compact=true")
                assert started.wait(5)
                token = store.memory.entries["10000001"].token
                client.get("/api/patients/10000002?compact=true")
                client.get("/api/patients/10000001?compact=true")
                assert not store.memory.current("10000001", token)
            finally:
                release.set()
            assert finished.wait(5)
            refreshed = await_patient(client)
            assert all(r["hadm_id"] != "99999999" for r in refreshed["admissions"])
            await_tables(client)
            result = client.get("/api/patients/10000001/tables/labevents").json()
            assert result["total"] == 6
            assert all(r["raw"]["value"] != "STALE" for r in result["rows"])


def test_preview_reuses_encoded_bytes_and_invalidates_changed_source(indexed):
    config, _ = indexed
    with TestClient(create_app(config)) as client:
        await_patient(client)
        url = "/api/images/50000001-image?size=128&format=webp"
        from mimic_atlas.backend import images

        with patch.object(images.Image, "open", wraps=images.Image.open) as opened:
            first = client.get(url)
            assert first.status_code == 200
            assert client.get(url).content == first.content and opened.call_count == 1
            path = client.app.state.store.images["50000001-image"].image_path
            Image.new("L", (48, 64), 250).save(path)
            changed = client.get(url)
            assert changed.content != first.content and opened.call_count == 2
        stats = client.get("/api/memory").json()["images"]
        assert stats["hits"] >= 1 and stats["bytes"] <= stats["max_bytes"]
