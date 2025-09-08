import os
from collections import defaultdict

from datasets.base_dataset import Datum, DatasetBase, get_dataset_info
from datasets.build_dataset import DATASET_REGISTRY
from datasets.dg.tiger import Tiger
from datasets.dg.stoat import Stoat
from datasets.dg.kiwi import Kiwi
from collections import defaultdict


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
            TARGET_DOMAINS: ["market1501"]
    
    Domain label assignment:
        - tiger → domain_label = 0
        - stoat → domain_label = 1  
        - market1501 → domain_label = 2
    """
    
    def __init__(self, cfg, verbose=True):
        self.cfg = cfg
        self._dataset_dir = "MultiReID"
        
        # Available dataset classes mapping
        self.dataset_classes = {
            "tiger": Tiger,
            "stoat": Stoat,
            "kiwi": Kiwi
        }
        
        # Combine all domains
        all_domains = set()
        if hasattr(cfg.DATASET, 'SOURCE_DOMAINS'):
            all_domains.update(cfg.DATASET.SOURCE_DOMAINS)
        if hasattr(cfg.DATASET, 'TARGET_DOMAINS'):
            all_domains.update(cfg.DATASET.TARGET_DOMAINS)

        self.all_domains = list(all_domains)

        

        # Load datasets and combine data
        train_data = []
        gallery_data = []
        query_data = []
        
        for domain_name in all_domains:
            if domain_name not in self.dataset_classes:
                raise ValueError(f"Dataset '{domain_name}' not supported. Available: {list(self.dataset_classes.keys())}")
            
            # Create individual dataset
            dataset_class = self.dataset_classes[domain_name]
            dataset = dataset_class(cfg, domain_label=self._get_domain_label_for_dataset(domain_name), verbose=False)

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
        
        # Build static per-domain ID offsets for globally unique labels
        domain_to_ids = defaultdict(set)
        for d in self._trains_data:
            # Each "d" is a Datum with fields aid (local id) and domain_label (int)
            domain_to_ids[d.domain_label].add(d.aid)

        self.domain_label_offsets = {}
        running_offset = 0
        for dom in sorted(domain_to_ids.keys()):
            self.domain_label_offsets[dom] = running_offset
            running_offset += len(domain_to_ids[dom])

        # Expose the total number of globally unique classes (train-time)
        self.num_classes = running_offset

        # Calculate statistics
        self.num_train_imgs, self.num_train_aids, self.num_train_cams, self.num_train_views = get_dataset_info(self.train_data)
        self.num_gallery_imgs, self.num_gallery_aids, self.num_gallery_cams, self.num_gallery_views = get_dataset_info(self.gallery_data)
        self.num_query_imgs, self.num_query_aids, self.num_query_cams, self.num_query_views = get_dataset_info(self.query_data)
    
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
