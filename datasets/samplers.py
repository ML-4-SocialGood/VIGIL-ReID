import copy
import numpy as np
import random

from collections import defaultdict
from torch.utils.data.sampler import Sampler


def build_sampler(data_source, sampler_type, batch_size, num_instances=None):
    assert isinstance(sampler_type, str)
    assert isinstance(batch_size, int)
    # assert isinstance(num_instances, int)
    
    if sampler_type == "RandomIdentitySampler":
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)
    elif sampler_type == "RandomSampler":
        sampler = Sampler.RandomSampler(data_source)
    elif sampler_type == "SequentialSampler":
        sampler = Sampler.SequentialSampler(data_source)
    return sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        """
        Randomly sample N identities, then for each identity, 
        randomly sample K instances, therefore the batch size is N*K.

        Args:
            data_source (list): A list of (img_path, aid, camid, viewid, domain).
            batch_size (int): The number of samples in a batch.
            num_instances (int): The number of instances per identity in a batch.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_aids_per_batch = self.batch_size // self.num_instances
        self.index_dict = defaultdict(list)    # a dict with integer keys and list values (e.g., {aid:[indices]})
                                               # {327: [0, 6, 50, 118, 641, 1022], ...}
        for index, (_, aid, _, _) in enumerate(self.data_source):
            self.index_dict[aid].append(index)
        self.aids = list(self.index_dict.keys())    # a list of all IDs

        # Estimate number of samples in an epoch. 
        self.length = 0
        for aid in self.aids:
            idxs = self.index_dict[aid]    # a list of indices for a given ID
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
        self.length = (self.length // self.batch_size) * self.batch_size    # ensure the length is a multiple of batch size

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)    # {aid: [[], [], ...]}

        for aid in self.aids:
            idxs = copy.deepcopy(self.index_dict[aid])    # a list of indices for a given ID
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size = self.num_instances, replace = True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[aid].append(batch_idxs)
                    batch_idxs = []

        avai_aids = copy.deepcopy(self.aids)    # a list of available IDs
        final_idxs = []

        while len(avai_aids) >= self.num_aids_per_batch:
            selected_aids = random.sample(avai_aids, self.num_aids_per_batch)
            for aid in selected_aids:
                batch_idxs = batch_idxs_dict[aid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[aid]) == 0:
                    avai_aids.remove(aid)

        return iter(final_idxs)

    def __len__(self):
        return self.length




random.seed(0)
data = [(f"img_{i}.jpg", i // 5, 0, 0) for i in range(15)]
print(data)
print()
sampler = build_sampler(data_source = data, sampler_type = "RandomIdentitySampler", batch_size = 8, num_instances = 4)
print(len(sampler))
print(list(iter(sampler)))
print(len(list(iter(sampler))))