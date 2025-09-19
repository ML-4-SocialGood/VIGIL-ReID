from utils import Registry, check_availability

DATASET_REGISTRY = Registry("DATASET")

from datasets.reid.multi_reid import MultiReID

def build_dataset(cfg):
    """
        Builds a list of datasets, each corresponding to a domain.
    """
    available_datasets = DATASET_REGISTRY.registered_names()
        
    # Combine all domains
    all_domains = set()
    if hasattr(cfg.DATASET, 'SOURCE_DOMAINS'):
        all_domains.update(cfg.DATASET.SOURCE_DOMAINS)
    else:
        raise ValueError("SOURCE_DOMAINS is not set in the config")
    if hasattr(cfg.DATASET, 'TARGET_DOMAINS'):
        all_domains.update(cfg.DATASET.TARGET_DOMAINS)
    else:
        raise ValueError("TARGET_DOMAINS is not set in the config")

    all_domains = list(all_domains)
    list_of_datasets = []
    for index, domain_name in enumerate(all_domains):
        check_availability(domain_name, available_datasets)

        # Create individual dataset instance
        dataset = DATASET_REGISTRY.get(domain_name)(cfg, domain_label=index, verbose=False)
        list_of_datasets.append(dataset)


    return list_of_datasets
