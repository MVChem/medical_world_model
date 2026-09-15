"""Independent deadline and orphan cleanup for this experiment's child sessions.

Every training command starts a new session. A session ID stays allocated while
members exist, including after its original leader exits. PID start times guard
individual signals, and a reused leader PID invalidates the old session claim.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import time


def proc_info(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), parent=int(fields[1]), group=int(fields[2]), session=int(fields[3]), start=fields[19])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def processes():
    rows = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row = proc_info(int(path.name))
            if row:
                rows.append(row)
    return rows


def members(record, rows):
    leader, start = record.get('child_pid'), record.get('child_start')
    if not leader or not start:
        return []
    current = proc_info(leader)
    if current and current['start'] != start:
        return []  # PID reused by a later, unrelated process/session.
    return [row for row in rows if row['session'] == leader and int(row['start']) >= int(start)]


def signal_member(row, sig):
    current = proc_info(row['pid'])
    if current and (current['start'], current['session']) == (row['start'], row['session']):
        try:
            os.kill(row['pid'], sig)
            return True
        except ProcessLookupError:
            pass
    return False


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scheduler', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--soft-deadline', default='2026-09-14T07:45:00+08:00')
    p.add_argument('--hard-deadline', default='2026-09-14T08:00:00+08:00')
    args = p.parse_args()
    soft = dt.datetime.fromisoformat(args.soft_deadline).timestamp()
    hard = dt.datetime.fromisoformat(args.hard_deadline).timestamp()
    args.out.mkdir(parents=True, exist_ok=True)
    sessions, orphan_since, actions = {}, {}, []
    while True:
        now = time.time()
        for scheduler in args.scheduler:
            for path in scheduler.glob('attempts/*/record.json'):
                record = read(path)
                if record.get('child_start'):
                    sessions[str(path)] = record
        rows = processes()
        active = []
        for path, record in sessions.items():
            owned = members(record, rows)
            if not owned:
                continue
            active.extend(row['pid'] for row in owned)
            leader = proc_info(record['child_pid'])
            leader_alive = leader and leader['start'] == record['child_start']
            if not leader_alive:
                orphan_since.setdefault(path, now)
            sig, reason = None, None
            # Give CUDA ten seconds to release contexts by the requested cutoff.
            if now >= hard - 10:
                sig, reason = signal.SIGKILL, 'hard deadline'
            elif not leader_alive:
                sig = signal.SIGKILL if now - orphan_since[path] >= 5 else signal.SIGTERM
                reason = 'descendant remained after training leader exited'
            elif now >= soft:
                sig, reason = signal.SIGTERM, 'soft deadline'
            if sig:
                for row in owned:
                    if signal_member(row, sig):
                        action = dict(time=now, attempt_id=record['attempt_id'], pid=row['pid'], signal=int(sig), reason=reason)
                        actions.append(action)
                        with (args.out / 'guardian_actions.jsonl').open('a') as f:
                            f.write(json.dumps(action) + '\n')
        value = dict(pid=os.getpid(), updated=now, known_attempts=len(sessions), active_owned_pids=active,
                     soft_deadline=args.soft_deadline, hard_deadline=args.hard_deadline,
                     recent_actions=actions[-20:], complete=now >= hard and not active)
        temp = args.out / f'guardian.{os.getpid()}.tmp'
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(args.out / 'guardian.json')
        if now >= hard and not active:
            return
        time.sleep(min(2, max(.1, hard - now)))


if __name__ == '__main__':
    main()
