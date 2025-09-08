from utils import Registry, check_availability
from datasets.dg.multi_reid import MultiReID

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(cfg):
    """
        Builds the multi-species dataset instance based on the configuration. Each dataset treated as a domain.

    """
    return MultiReID(cfg)
