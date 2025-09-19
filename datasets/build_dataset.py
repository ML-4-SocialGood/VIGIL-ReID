from utils import Registry

DATASET_REGISTRY = Registry("DATASET")

from datasets.reid.multi_reid import MultiReID

def build_dataset(cfg):
    """
        Builds the multi-species dataset instance based on the configuration. Each dataset treated as a domain.

    """
    return MultiReID(cfg)
