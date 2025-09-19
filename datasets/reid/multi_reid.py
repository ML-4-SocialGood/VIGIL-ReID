import os
from collections import defaultdict

from datasets.base_dataset import Datum, DatasetBase, get_dataset_info
from datasets.build_dataset import DATASET_REGISTRY
from datasets.reid.tiger import Tiger
from datasets.reid.stoat import Stoat
from datasets.reid.kiwi import Kiwi
from collections import defaultdict
from utils import check_availability

# this class is no longer used
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
        
        self.num_classes, self.train_global_ids = self.convert_to_global_id(self.train_data)
        _, self.test_global_ids = self.convert_to_global_id(self.query_data + self.gallery_data)

        # Sorted list for stable indexing/order
        self.class_names = sorted(self.train_global_ids)


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
