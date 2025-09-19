import os
from collections import defaultdict

from datasets.base_dataset import Datum, DatasetBase, get_dataset_info
from datasets.build_dataset import DATASET_REGISTRY
from datasets.reid.tiger import Tiger
from datasets.reid.stoat import Stoat
from datasets.reid.kiwi import Kiwi
from collections import defaultdict
from utils import check_availability

@DATASET_REGISTRY.register()
class MultiReID(DatasetBase):
    """
    Multi-Dataset ReID wrapper that combines multiple ReID datasets for domain adaptation.
    
    Each dataset is treated as a domain, with automatic domain label assignment based on 
    the SOURCE_DOMAINS and TARGET_DOMAINS configuration.
    
    Example configuration:
        DATASET:
            NAME: "MultiReID"
            SOURCE_DOMAINS: ["tiger", "stoat"] 
            TARGET_DOMAINS: ["stoat", "kiwi"]
    
    Domain label assignment:
        - tiger → domain_label = 0
        - stoat → domain_label = 1  
        - kiwi → domain_label = 2
    """
    
    def __init__(self, cfg, verbose=True):
        self.cfg = cfg
        self._dataset_dir = "MultiReID"
        
        available_datasets = DATASET_REGISTRY.registered_names()
        
        # Combine all domains
        all_domains = set()
        if hasattr(cfg.DATASET, 'SOURCE_DOMAINS'):
            all_domains.update(cfg.DATASET.SOURCE_DOMAINS)
        if hasattr(cfg.DATASET, 'TARGET_DOMAINS'):
            all_domains.update(cfg.DATASET.TARGET_DOMAINS)

        self.all_domains = list(all_domains)
        self.source_domains = cfg.DATASET.SOURCE_DOMAINS if hasattr(cfg.DATASET, 'SOURCE_DOMAINS') else []
        self.target_domains = cfg.DATASET.TARGET_DOMAINS if hasattr(cfg.DATASET, 'TARGET_DOMAINS') else []
        

        # Load datasets and combine data
        train_data = []
        gallery_data = []
        query_data = []
        
        for domain_name in all_domains:
            check_availability(domain_name, available_datasets)

            # Create individual dataset instance
            dataset = DATASET_REGISTRY.get(domain_name)(cfg, domain_label=self._get_domain_label_for_dataset(domain_name), verbose=False)

            # Combine the data, test consists of gallery + query
            if domain_name in cfg.DATASET.SOURCE_DOMAINS:
                train_data.extend(dataset.train_data)
            if domain_name in cfg.DATASET.TARGET_DOMAINS:
                gallery_data.extend(dataset.gallery_data)
                query_data.extend(dataset.query_data)

        # Initialize base class
        super().__init__(
            dataset_dir=self._dataset_dir,
            domain="multi_reid",  # Multi-dataset identifier
            train_data=train_data,
            gallery_data=gallery_data, 
            query_data=query_data
        )
        
        if verbose:
            print("=> MultiReID loaded")
            self.show_dataset_info()
        
        self.num_classes, self.train_global_ids = self.convert_to_global_id(self.train_data)
        _, self.test_global_ids = self.convert_to_global_id(self.query_data + self.gallery_data)

        # Sorted list for stable indexing/order
        self.class_names = sorted(self.train_global_ids)


        # Calculate statistics
        self.num_train_imgs, self.num_train_aids, self.num_train_cams, self.num_train_views = get_dataset_info(self.train_data)
        self.num_gallery_imgs, self.num_gallery_aids, self.num_gallery_cams, self.num_gallery_views = get_dataset_info(self.gallery_data)
        self.num_query_imgs, self.num_query_aids, self.num_query_cams, self.num_query_views = get_dataset_info(self.query_data)
    
    def convert_to_global_id(self, dataset):
        domain_to_ids = defaultdict(set)
        for d in dataset:
            # Each "d" is a Datum with fields aid (local id) and domain_label (int)
            domain_to_ids[d.domain_label].add(d.aid)

        domain_label_offsets = {}
        running_offset = 0
        for dom in sorted(domain_to_ids.keys()):
            domain_label_offsets[dom] = running_offset
            running_offset += len(domain_to_ids[dom])

        # Expose the total number of globally unique classes (train-time)
        num_classes = running_offset

        # Build globally unique class names using domain-specific offsets
        # Each global class id = local aid + offset(domain_label)
        global_ids = set()
        for d in dataset:
            global_id = domain_label_offsets[d.domain_label] + d.aid
            d.aid = global_id
            global_ids.add(global_id)
        return num_classes, global_ids
    
    def _get_domain_label_for_dataset(self, dataset_name):
        """
        Get domain label for a specific dataset based on configuration.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            int: Domain label for this dataset
        """
        # Check if in source domains
        if dataset_name in self.all_domains:
            return self.all_domains.index(dataset_name)
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found in configured domains.")
    
    def show_domain_mapping(self):
        """Display the domain to label mapping for debugging."""
        print("\nDomain Label Mapping:")
        print("-" * 30)
        
        if hasattr(self.cfg.DATASET, 'SOURCE_DOMAINS'):
            for i, domain in enumerate(self.cfg.DATASET.SOURCE_DOMAINS):
                print(f"{domain:>15} → domain_label = {i} (source)")
        
        if hasattr(self.cfg.DATASET, 'TARGET_DOMAINS'):
            source_count = len(self.cfg.DATASET.SOURCE_DOMAINS) if hasattr(self.cfg.DATASET, 'SOURCE_DOMAINS') else 0
            for i, domain in enumerate(self.cfg.DATASET.TARGET_DOMAINS):
                print(f"{domain:>15} → domain_label = {source_count + i} (target)")
        
        print("-" * 30)
