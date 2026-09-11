"""Opt-in process-local workaround for vLLM's all-physical-GPU name warning.

This host has an unavailable GPU 4. vLLM 0.22.1 enumerates its name even when
CUDA_VISIBLE_DEVICES excludes it. Skip ONLY that advisory name scan; retain
real counts, device selection, memory checks and errors everywhere else.
No files in the shared Python environment are modified.
"""
import os
import sys

if os.environ.get('MIMIC_VLLM_SKIP_PHYSICAL_NAME_WARNING') == '1':
    import vllm.third_party.pynvml as _nvml
    _real_count = _nvml.nvmlDeviceGetCount

    def _count_for_warning():
        frame = sys._getframe(1)
        if (frame.f_code.co_name == 'log_warnings'
                and frame.f_globals.get('__name__') == 'vllm.platforms.cuda'):
            return 1
        return _real_count()

    _nvml.nvmlDeviceGetCount = _count_for_warning

if os.environ.get('MIMIC_VLLM_UUID_DEVICES') == '1':
    # CUDA excludes the failed device and renumbers integer ordinals on this
    # host. UUIDs select the intended physical GPU without that ambiguity.
    import vllm.third_party.pynvml as _nvml
    from vllm.platforms.interface import Platform
    _original_physical_id = Platform.device_id_to_physical_device_id.__func__

    def _physical_id(cls, device_id):
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        if device_id<len(visible) and visible[device_id].startswith('GPU-'):
            _nvml.nvmlInit()
            try:
                handle=_nvml.nvmlDeviceGetHandleByUUID(visible[device_id])
                return _nvml.nvmlDeviceGetIndex(handle)
            finally:
                _nvml.nvmlShutdown()
        return _original_physical_id(cls,device_id)

    Platform.device_id_to_physical_device_id=classmethod(_physical_id)

if os.environ.get('MIMIC_VLLM_PYTORCH_TOPK') == '1':
    # The 248k-token Qwen vocabulary causes pathological Triton sort compilation
    # in this installed vLLM's >=8-sequence sampler warmup. Use the same module's
    # existing PyTorch implementation. The actual screening is greedy (T=0).
    from vllm.v1.sample.ops import topk_topp_sampler as _sampling
    _sampling.apply_top_k_top_p = _sampling.apply_top_k_top_p_pytorch

if os.environ.get('MIMIC_VLLM_STACK_SIGNAL') == '1':
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1,all_threads=True)
