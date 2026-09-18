"""Bounded process-memory caches; no patient or image payload is written to disk."""

from __future__ import annotations

import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field


class CacheBusy(RuntimeError):
    """All cache slots are currently leased by requests."""


class DiscardedLoad(RuntimeError):
    """The patient was evicted while background work was in progress."""


def deep_size(value):
    """Estimate owned Python containers once when publishing, counting aliases once."""
    seen, pending, total = set(), [value], 0
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending.extend(item)
    return total


@dataclass
class PatientEntry:
    touched: float
    token: object = field(default_factory=object)
    pins: int = 0
    sizes: dict = field(default_factory=dict)


class PatientMemory:
    """One LRU/TTL policy for every part of a patient's cached data."""

    def __init__(
        self,
        lock,
        on_evict,
        *,
        max_patients=4,
        max_bytes=512 * 2**20,
        ttl=300,
        clock=time.monotonic,
    ):
        if max_patients < 1 or max_bytes < 1 or ttl <= 0:
            raise ValueError("Patient cache limits must be positive")
        self.lock, self.on_evict, self.clock = lock, on_evict, clock
        self.max_patients, self.max_bytes, self.ttl = max_patients, max_bytes, ttl
        self.entries = OrderedDict()
        self.evictions = 0

    def _drop(self, subject):
        entry = self.entries.pop(subject)
        self.on_evict(subject, entry.token)
        self.evictions += 1

    def sweep(self):
        with self.lock:
            now = self.clock()
            for subject, entry in list(self.entries.items()):
                if not entry.pins and now - entry.touched >= self.ttl:
                    self._drop(subject)
            # Keep a single large patient's complete records, rather than an endless
            # load/evict/reload loop. In-flight response leases are also protected.
            while (
                len(self.entries) > 1
                and sum(sum(e.sizes.values()) for e in self.entries.values())
                > self.max_bytes
            ):
                victim = next(
                    (
                        s
                        for s, e in self.entries.items()
                        if not e.pins and s != next(reversed(self.entries))
                    ),
                    None,
                )
                if victim is None:
                    break
                self._drop(victim)

    def touch(self, subject):
        with self.lock:
            self.sweep()
            if subject not in self.entries:
                while len(self.entries) >= self.max_patients:
                    victim = next(
                        (s for s, e in self.entries.items() if not e.pins), None
                    )
                    if victim is None:
                        raise CacheBusy("Patient requests are busy; retry shortly")
                    self._drop(victim)
                self.entries[subject] = PatientEntry(self.clock())
            entry = self.entries[subject]
            entry.touched = self.clock()
            self.entries.move_to_end(subject)
            return entry.token

    def current(self, subject, token):
        with self.lock:
            entry = self.entries.get(subject)
            return entry is not None and entry.token is token

    def account(self, subject, token, part, size):
        with self.lock:
            if self.current(subject, token):
                self.entries[subject].sizes[part] = size
                self.sweep()

    @contextmanager
    def lease(self, subject):
        with self.lock:
            token = self.touch(subject)
            self.entries[subject].pins += 1
        try:
            yield token
        finally:
            with self.lock:
                if self.current(subject, token):
                    self.entries[subject].pins -= 1
                self.sweep()

    def clear(self):
        with self.lock:
            for subject in list(self.entries):
                self._drop(subject)

    def stats(self):
        with self.lock:
            return {
                "patients": len(self.entries),
                "max_patients": self.max_patients,
                "estimated_bytes": sum(
                    sum(e.sizes.values()) for e in self.entries.values()
                ),
                "budget_bytes": self.max_bytes,
                "idle_seconds": self.ttl,
                "evictions": self.evictions,
                "leased": sum(e.pins for e in self.entries.values()),
                "subjects": list(self.entries),
            }


class ImageMemory:
    """LRU/TTL of encoded thumbnails with exact byte and entry limits."""

    def __init__(
        self, *, max_bytes=64 * 2**20, max_entries=512, ttl=300, clock=time.monotonic
    ):
        if max_bytes < 0 or max_entries < 1 or ttl <= 0:
            raise ValueError("Invalid image cache limits")
        self.max_bytes, self.max_entries, self.ttl, self.clock = (
            max_bytes,
            max_entries,
            ttl,
            clock,
        )
        self.lock = threading.RLock()
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0
        self.render_slots = threading.BoundedSemaphore(2)
        self.key_locks = [threading.Lock() for _ in range(32)]

    def _drop(self, key):
        payload, _ = self.entries.pop(key)
        self.bytes -= len(payload)
        self.evictions += 1

    def sweep(self):
        with self.lock:
            for key, (_, touched) in list(self.entries.items()):
                if self.clock() - touched >= self.ttl:
                    self._drop(key)

    def get(self, key):
        with self.lock:
            self.sweep()
            entry = self.entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            payload, _ = entry
            self.entries[key] = (payload, self.clock())
            self.entries.move_to_end(key)
            self.hits += 1
            return payload

    def put(self, key, payload):
        with self.lock:
            self.sweep()
            if key in self.entries:
                self._drop(key)
            if len(payload) > self.max_bytes:
                return
            while self.entries and (
                len(self.entries) >= self.max_entries
                or self.bytes + len(payload) > self.max_bytes
            ):
                self._drop(next(iter(self.entries)))
            self.entries[key] = (payload, self.clock())
            self.bytes += len(payload)

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.bytes = 0

    def stats(self):
        with self.lock:
            return {
                "entries": len(self.entries),
                "max_entries": self.max_entries,
                "bytes": self.bytes,
                "max_bytes": self.max_bytes,
                "idle_seconds": self.ttl,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "max_concurrent_decodes": 2,
            }
