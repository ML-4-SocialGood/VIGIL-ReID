import torch
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tabulate import tabulate
from .build_dataset import build_dataset
from .samplers import build_sampler
from .transforms import build_transform


def train_collate_fn(batch):
    """Collate function for training data loader."""
    batch_dict = {}
    img_paths, aids, camids, viewids, imgs, domains = [], [], [], [], [], []
    for b_dict in batch:
        img_paths.append(b_dict["img_path"])
        aids.append(b_dict["aid"])
        camids.append(b_dict["camid"])
        viewids.append(b_dict["viewid"])
        imgs.append(b_dict["img"])
        domains.append(b_dict["domain"]) 
    
    batch_dict["img_paths"] = img_paths
    batch_dict["aids"] = torch.tensor(aids, dtype = torch.int64)
    batch_dict["camids"] = camids
    batch_dict["viewids"] = torch.tensor(viewids, dtype = torch.int64)
    batch_dict["imgs"] = torch.stack(imgs, dim = 0)
    batch_dict["domains"] = domains 

    return batch_dict


def test_collate_fn(batch):
    """Collate function for test data loader."""
    batch_dict = {}
    img_paths, aids, camids, viewids, imgs, domains = [], [], [], [], [], []
    for b_dict in batch:
        img_paths.append(b_dict["img_path"])
        aids.append(b_dict["aid"])
        camids.append(b_dict["camid"])
        viewids.append(b_dict["viewid"])
        imgs.append(b_dict["img"])
        domains.append(b_dict["domain"])  
    
    batch_dict["img_paths"] = img_paths
    batch_dict["aids"] = torch.tensor(aids, dtype = torch.int64)
    batch_dict["camids"] = camids
    batch_dict["viewids"] = torch.tensor(viewids, dtype = torch.int64)
    batch_dict["imgs"] = torch.stack(imgs, dim = 0)
    batch_dict["domains"] = domains  

    return batch_dict


def build_data_loader(
        cfg, 
        sampler_type = "RandomIdentitySampler", 
        data_source = None, 
        batch_size = 32, 
        transform = None, 
        is_train = True,
        domain_label_offsets = None
):
    if is_train:
        sampler = build_sampler(data_source = data_source, 
                                sampler_type = sampler_type, 
                                batch_size = batch_size, 
                                num_instances = cfg.DATALOADER.TRAIN.NUM_INSTANCES)
        data_loader = DataLoader(
            dataset = DatasetWrapper(cfg = cfg, data_source = data_source, transform = transform, is_train = is_train, domain_label_offsets = domain_label_offsets), 
            batch_size = batch_size, 
            sampler = sampler, 
            num_workers = cfg.DATALOADER.NUM_WORKERS, 
            collate_fn = train_collate_fn
        )
    else:
        data_loader = DataLoader(
            dataset = DatasetWrapper(cfg = cfg, data_source = data_source, transform = transform, is_train = is_train, domain_label_offsets = domain_label_offsets), 
            batch_size = batch_size, 
            num_workers = cfg.DATALOADER.NUM_WORKERS, 
            collate_fn = test_collate_fn,
            shuffle=False
        )
    
    assert len(data_loader) > 0, "Empty data loader"
    return data_loader


class DataManager:
    def __init__(self, cfg):
        self.dataset = build_dataset(cfg)    # initialize multi-domain dataset, each domain is a dataset (eg. Kiwi, Tiger, Stoat)

        transform_train = build_transform(cfg, is_train = True)
        transform_test = build_transform(cfg, is_train = False)
        
        self.data_loader_train = build_data_loader(
            cfg = cfg, 
            sampler_type = cfg.DATALOADER.TRAIN.SAMPLER, 
            data_source = self.dataset.train_data, 
            batch_size = cfg.DATALOADER.TRAIN.BATCH_SIZE, 
            transform = transform_train, 
            is_train = True,
            domain_label_offsets = getattr(self.dataset, 'domain_label_offsets', None)
        )

        self.data_loader_test = build_data_loader(
            cfg = cfg, 
            sampler_type = cfg.DATALOADER.TEST.SAMPLER, 
            data_source = self.dataset.query_data + self.dataset.gallery_data, 
            batch_size = cfg.DATALOADER.TEST.BATCH_SIZE, 
            transform = transform_test, 
            is_train = False,
            domain_label_offsets = getattr(self.dataset, 'domain_label_offsets', None)
        )

        # Num of animal IDs in the training set. If the same animal ID appears in different datasets, it will be counted multiple times as they come from different domains.
        self._num_classes = getattr(self.dataset, 'num_classes', None)
        self.class_names = getattr(self.dataset, 'class_names', None)
        
        self.len_query = len(self.dataset.query_data)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        if hasattr(self.dataset, 'all_domains'):
            return len(self.dataset.all_domains)
        return 0

    @property
    def get_num_train_images(self):
        return self.dataset.num_train_imgs
    
    @property
    def get_num_train_aids(self):
        return self.dataset.num_train_aids
    
    @property
    def get_num_train_camids(self):
        return self.dataset.num_train_cams
    
    @property
    def get_num_train_viewids(self):
        return self.dataset.num_train_views
    
    @property
    def get_num_gallery_images(self):
        return self.dataset.num_gallery_imgs
    
    @property
    def get_num_gallery_aids(self):
        return self.dataset.num_gallery_aids
    
    @property
    def get_num_gallery_camids(self):
        return self.dataset.num_gallery_cams
    
    @property
    def get_num_gallery_viewids(self):
        return self.dataset.num_gallery_views
    
    @property
    def get_num_query_images(self):
        return self.dataset.num_query_imgs
    
    @property
    def get_num_query_aids(self):
        return self.dataset.num_query_aids
    
    @property
    def get_num_query_camids(self):
        return self.dataset.num_query_cams
    
    @property
    def get_num_query_viewids(self):
        return self.dataset.num_query_views


class DatasetWrapper(Dataset):
    def __init__(self, cfg, data_source, transform = None, is_train = False, domain_label_offsets = None):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train
        self.domain_label_offsets = domain_label_offsets or {}
    
    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, idx):
        datum = self.data_source[idx]

        output = {
            "img_path": datum.img_path, 
            # Remap local id to globally unique id using domain-specific offsets
            "aid": (datum.aid + self.domain_label_offsets.get(datum.domain_label, 0)), 
            "camid": datum.camid, 
            "viewid": datum.viewid, 
            "domain": datum.domain_label 
        }

        img = Image.open(datum.img_path).convert("RGB")
        img = self.transform(img)
        output["img"] = img

        return output