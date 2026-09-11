"""Launch an owned, localhost-only Qwen replica on an explicitly selected free GPU."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from common import dump_json


def launch(a):
    os.umask(0o077)
    a.out.mkdir(parents=True,exist_ok=True)
    # Check exactly the requested physical device; do not enumerate the broken GPU.
    import vllm.third_party.pynvml as nvml
    nvml.nvmlInit()
    try:
        handle=nvml.nvmlDeviceGetHandleByIndex(a.gpu)
        memory=nvml.nvmlDeviceGetMemoryInfo(handle)
        util=nvml.nvmlDeviceGetUtilizationRates(handle)
        if memory.used>512*1024**2 or util.gpu>5:
            raise RuntimeError(f'GPU {a.gpu} is not free: {memory.used/1024**2:.0f} MiB, {util.gpu}% activity')
        uuid=nvml.nvmlDeviceGetUUID(handle)
    finally: nvml.nvmlShutdown()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',a.port))
    args=[sys.executable,'-m','vllm.entrypoints.cli.main','serve',str(a.model),
        '--served-model-name','mimic-qwen35-9b','--host','127.0.0.1','--port',str(a.port),
        '--tensor-parallel-size','1','--dtype','bfloat16','--max-model-len','8192',
        '--max-num-seqs','16','--max-num-batched-tokens','4096','--gpu-memory-utilization','0.94',
        '--limit-mm-per-prompt','{"image":2,"video":0}',
        '--mm-processor-kwargs','{"size":{"longest_edge":1048576,"shortest_edge":262144}}',
        '--mm-encoder-tp-mode','data','--allowed-local-media-path',str(a.cxr),
        '--reasoning-parser','qwen3','--no-enable-log-requests','--disable-uvicorn-access-log',
        '--no-enable-prefix-caching']
    env=os.environ.copy()
    updates=dict(CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',MIMIC_VLLM_UUID_DEVICES='1',
        MIMIC_VLLM_SKIP_PHYSICAL_NAME_WARNING='1',PYTHONPATH=str(Path(__file__).parent.resolve()/'vlm_runtime'),
        HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',VLLM_NO_USAGE_STATS='1',
        OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',
        VLLM_USE_FLASHINFER_SAMPLER='0',MIMIC_VLLM_PYTORCH_TOPK='1',MIMIC_VLLM_STACK_SIGNAL='1')
    env.update(updates)
    with (a.out/f'server_{a.port}.log').open('a') as f:
        proc=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    dump_json(a.out/f'server_{a.port}.json',dict(pid=proc.pid,args=args,env=updates,gpu_uuid=uuid,
        proc_start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21]))
    print(f'GPU {a.gpu}, localhost:{a.port}, PID {proc.pid}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/home/data2/chk/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'))
    p.add_argument('--cxr',type=Path,default=Path('/home/data1/data/MIMIC/MIMIC_CXR'))
    launch(p.parse_args())
