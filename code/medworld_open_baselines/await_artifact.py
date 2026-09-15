"""Wait for another owned preparation process without reserving a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path',type=Path,required=True)
    p.add_argument('--additional-path',type=Path,action='append',default=[])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fields',default='{}')
    p.add_argument('--timeout',type=float,default=86400)
    p.add_argument('--producer-launch',type=Path)
    args=p.parse_args();expected=json.loads(args.fields);start=time.monotonic()
    while True:
        ready=True
        for path in [args.path]+args.additional_path:
            if not path.exists():ready=False;continue
            data=json.loads(path.read_text())
            if data.get('status') in ['failed','blocked']:
                launch=json.loads(args.producer_launch.read_text()) if args.producer_launch else {}
                if not args.producer_launch or data.get('finished',0)>=launch.get('started',0):
                    raise RuntimeError(str(data.get('error_type',data.get('status'))))
            if not all(data.get(k)==v for k,v in expected.items()):ready=False
        if ready:break
        if time.monotonic()-start>=args.timeout:raise TimeoutError(str(args.path))
        time.sleep(10)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(status='complete',path=str(args.path),
        sha256=hashlib.sha256(args.path.read_bytes()).hexdigest(),
        additional_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.additional_path},
        checked_fields=expected,finished=time.time()),indent=2)+'\n')

if __name__=='__main__':main()
