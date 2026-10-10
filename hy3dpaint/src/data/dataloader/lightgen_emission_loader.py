# The emission loader is upstream's TextureDataset, edited in place (REFACTOR_TODO.md step 8a). This
# module keeps the class path that stored configs and the parent's predictor name; step 15 removes it.
from .objaverse_loader_forTexturePBR import TextureDataset as LightgenEmissionDataset  # noqa: F401
