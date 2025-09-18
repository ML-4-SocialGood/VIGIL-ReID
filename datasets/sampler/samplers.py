import copy
import numpy as np
import random

from collections import defaultdict
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from RandomIdentitySampler import RandomIdentitySampler
from MultiDomainRandomIdentitySampler import MultiDomainRandomIdentitySampler


def build_sampler(data_source, sampler_type, batch_size, num_instances=None, source_domains=None):
    assert isinstance(sampler_type, str)
    assert isinstance(batch_size, int)
    # assert isinstance(num_instances, int)
    
    if source_domains is not None:
        num_domains = len(source_domains)
    else:
        num_domains = 1
        
    if sampler_type == "RandomIdentitySampler" and num_domains == 1:
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)
    elif sampler_type == "RandomIdentitySampler" and num_domains > 1:
        sampler = MultiDomainRandomIdentitySampler(data_source, batch_size, num_instances)
    elif sampler_type == "RandomSampler":
        sampler = RandomSampler(data_source)
    elif sampler_type == "SequentialSampler":
        sampler = SequentialSampler(data_source)
    return sampler


