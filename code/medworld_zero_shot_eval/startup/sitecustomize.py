"""Process-local vLLM 0.22 GPU discovery fixes; never modifies shared packages.

The machine has an inaccessible GPU 4. vLLM's import-time warning enumerates all
physical devices despite CUDA_VISIBLE_DEVICES. It also expects numeric visible
device IDs, whereas our GPU lock uses the more reliable UUID. These two source
transformations affect discovery only, leaving kernels and model code unchanged.
Installed only in this queue's child PYTHONPATH, behind an explicit environment flag.
"""
import importlib.abc
import importlib.machinery
import os
import sys


PATCHES = {
    "vllm.platforms.cuda": (
        "    def log_warnings(cls):\n        device_ids: int = pynvml.nvmlDeviceGetCount()",
        "    def log_warnings(cls):\n        return  # queue already validated its single locked GPU\n        device_ids: int = pynvml.nvmlDeviceGetCount()",
    ),
    "vllm.platforms.interface": (
        "            physical_device_id = device_ids[device_id]\n            return int(physical_device_id)",
        "            physical_device_id = device_ids[device_id]\n"
        "            if physical_device_id.startswith('GPU-'):\n"
        "                from vllm.third_party import pynvml\n"
        "                pynvml.nvmlInit()\n"
        "                try:\n"
        "                    handle = pynvml.nvmlDeviceGetHandleByUUID(physical_device_id)\n"
        "                    return pynvml.nvmlDeviceGetIndex(handle)\n"
        "                finally:\n"
        "                    pynvml.nvmlShutdown()\n"
        "            return int(physical_device_id)",
    ),
}


class DiscoveryLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        text = self.get_data(self.path).decode("utf-8")
        old, new = PATCHES[fullname]
        if text.count(old) != 1:
            raise ImportError("vLLM discovery source changed; review the local compatibility adapter")
        return compile(text.replace(old, new), self.path, "exec")


class DiscoveryFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in PATCHES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            raise ImportError("Expected the inspected vLLM Python discovery source")
        spec.loader = DiscoveryLoader(fullname, spec.origin)
        return spec


if os.environ.get("MEDWORLD_VLLM_COMPAT") == "1":
    sys.meta_path.insert(0, DiscoveryFinder())
