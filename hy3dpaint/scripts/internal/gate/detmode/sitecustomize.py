import os
if os.environ.get("HUNYUAN_GATE_DETERMINISTIC") == "1":
    import sys
    import torch
    _strict = os.environ.get("HUNYUAN_GATE_DET_STRICT") == "1"   # raise at the first op with no deterministic kernel
    torch.use_deterministic_algorithms(True, warn_only=not _strict)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[detmode] torch.use_deterministic_algorithms(True, warn_only={not _strict}); cudnn.deterministic=True", file=sys.stderr, flush=True)
    if os.environ.get("HUNYUAN_GATE_MATH_SDP") == "1":
        # the fused attention kernels (flash, memory-efficient, cuDNN) are not flagged by deterministic
        # mode but their backward can sum in a varying order; the math kernel is deterministic
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        print("[detmode] scaled_dot_product_attention: math kernel only", file=sys.stderr, flush=True)
