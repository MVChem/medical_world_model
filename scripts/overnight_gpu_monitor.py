"""Record utilization of the five allocated GPUs until the overnight deadline."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--deadline', default='2026-09-14T08:00:00+08:00')
    p.add_argument('--gpus', default='0,1,2,3,4')
    args = p.parse_args()
    end = dt.datetime.fromisoformat(args.deadline).timestamp()
    allowed = {int(i) for i in args.gpus.split(',')}
    args.out.mkdir(parents=True, exist_ok=True)
    recent = []
    while time.time() < end:
        record = dict(time=time.time())
        try:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,power.draw',
                                            '--format=csv,noheader,nounits'], text=True, timeout=10)
            rows = []
            for line in raw.splitlines():
                idx, util, mem, power = [part.strip() for part in line.split(',')]
                if int(idx) in allowed:
                    rows.append(dict(gpu=int(idx), utilization=int(util), memory_mib=int(mem), power_w=float(power)))
            record['gpus'] = rows
            recent.append(record)
            recent = recent[-20:]
            averages = {str(i): round(sum(r['utilization'] for sample in recent for r in sample['gpus'] if r['gpu'] == i) /
                                     sum(r['gpu'] == i for sample in recent for r in sample['gpus']), 1)
                        for i in allowed if any(r['gpu'] == i for sample in recent for r in sample['gpus'])}
            latest = dict(**record, last_10min_mean_utilization=averages)
            temp = args.out / 'gpu_utilization.tmp.json'
            temp.write_text(json.dumps(latest, indent=2) + '\n')
            temp.replace(args.out / 'gpu_utilization.json')
        except Exception as exc:
            record['error'] = str(exc)
        with (args.out / 'gpu_utilization.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        time.sleep(min(30, max(0, end - time.time())))


if __name__ == '__main__':
    main()
