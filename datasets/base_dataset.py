import gdown
import os
import zipfile

from tabulate import tabulate


def get_dataset_info(dataset):
    aids_list, camids_list, viewids_list = [], [], []
    for entry in dataset:
        aids_list.append(entry.aid)
        camids_list.append(entry.camid)
        viewids_list.append(entry.viewid)

    aids = set(aids_list)
    camids = set(camids_list)
    viewids = set(viewids_list)
    
    num_of_images = len(dataset)
    num_of_aids = len(aids) # num of classes for animal ReID
    num_of_camids = len(camids)
    num_of_viewids = len(viewids)
    return num_of_images, num_of_aids, num_of_camids, num_of_viewids


class Datum:
    def __init__(self, img_path, aid, camid, viewid, domain_label):
        """
        Data instance for Animal ReID which defines the basic attributes.

        Args:
            img_path (str): Image path.
            aid (int): Animal ID.
            camid (str): Camera ID.
            viewid (int): View ID.
            domain_label (int): Domain label.
        """
        assert isinstance(img_path, str)
        assert os.path.isfile(img_path)
        assert isinstance(aid, int)
        assert isinstance(camid, str)
        assert isinstance(viewid, int)


        self._img_path = img_path
        self._aid = aid
        self._camid = camid
        self._viewid = viewid
        self._domain_label = domain_label

    @property
    def img_path(self):
        return self._img_path
    
    @property
    def aid(self):
        return self._aid
    
    @property
    def camid(self):
        return self._camid
    
    @property
    def viewid(self):
        return self._viewid
    
    @property
    def domain_label(self):
        return self._domain_label


class DatasetBase:
    def __init__(
        self, 
        dataset_dir, 
        domain,
        data_url=None, 
        train_data=None, 
        gallery_data=None, 
        query_data=None, 
    ):
        """
        Base class of Animal ReID dataset.

        Args:
            dataset_dir (str): Dataset directory.
            data_url (str): Dataset URL.
            train_data (list): A list of attributes of the training dataset.
            gallery_data (list): A list of attributes of the gallery dataset.
            query_data (list): A list of attributes of the query dataset.
            domain (string): Domain label of this dataset, same as the name of the Dataset.
            num_classes (int): Number of aniaml IDs in the training set.
            class_names (list): List of class names of the training set.
        """
        self._dataset_dir = dataset_dir
        self._data_url = data_url
        self._train_data = train_data
        self._gallery_data = gallery_data
        self._query_data = query_data
        self._domain = domain
        self._num_classes = self.get_num_classes()
        self.class_names = self.get_class_names()
    @property
    def dataset_dir(self):
        return self._dataset_dir
    
    @property
    def domain(self):
        return self._domain
    
    @property
    def data_url(self):
        return self._data_url
    
    @property
    def train_data(self):
        return self._train_data
    
    @property
    def gallery_data(self):
        return self._gallery_data
    
    @property
    def query_data(self):
        return self._query_data
    
    def download_data_from_gdrive(self, dst):
        gdown.download(self._data_url, dst, quiet=False)

        zip_ref = zipfile.ZipFile(dst, "r")
        zip_ref.extractall(os.path.dirname(dst))
        zip_ref.close()
        print("File Extracted to {}".format(os.path.dirname(dst)))
        os.remove(dst)

    def check_input_domains(self, source_domains, target_domain):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domain)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self._domains:
                raise ValueError(
                    "Input Domain Must Belong to {}, " "but Got [{}]".format(
                        self._domains, domain
                    )
                )
    
    def get_num_classes(self):
        return len([self._train_data[i].aid for i in range(len(self._train_data))])

    def get_class_names(self):
        return [self._train_data[i].aid for i in range(len(self._train_data))]

    def show_dataset_info(self):
        headers = ["Subset", "# images", "# ids", "# cameras"]
        train_info = get_dataset_info(self._train_data)
        gallery_info = get_dataset_info(self._gallery_data)
        query_info = get_dataset_info(self._query_data)
        table = [
            ["train"] + list(train_info)[0:3], 
            ["gallery"] + list(gallery_info)[0:3], 
            ["query"] + list(query_info)[0:3]
            ]
        print("Dataset statistics:")
        print(tabulate(table, headers = headers, tablefmt = "github"))
        


