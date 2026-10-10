# The emission model is upstream's HunyuanPaint, edited in place (REFACTOR_TODO.md step 8b). This
# module keeps the class path that stored configs name; step 15 removes it.
from .unet.model import HunyuanPaint as HunyuanPaintEmission  # noqa: F401
