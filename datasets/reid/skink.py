import os
from typing import Dict, List, Tuple

from datasets.base_dataset import DatasetBase, Datum, get_dataset_info
from datasets.build_dataset import DATASET_REGISTRY
from utils.tools import listdir_nonhidden


@DATASET_REGISTRY.register()
class Skink(DatasetBase):
    """
    TBD
    """

    def __init__(self, cfg, domain_label, verbose=True):
        self._dataset_dir = "Skink"
        root = cfg.DATASET.ROOT
        self._dataset_path = os.path.join(root, self._dataset_dir)
        self.domain_label = domain_label
        self._domain = "skink"

        self.train_dir = os.path.join(self._dataset_path, "train")
        self.gallery_dir = os.path.join(self._dataset_path, "gallery")
        self.query_dir = os.path.join(self._dataset_path, "query")
        self._check_before_run()

        train_data = self.read_data(data_dir=self.train_dir, relabel=False)
        # gallery_data = self.read_data(data_dir=self.gallery_dir, relabel=False)
        # query_data = self.read_data(data_dir=self.query_dir, relabel=False)
        query_data, gallery_data, aid2label = self.read_test_data(
            query_dir=self.query_dir, gallery_dir=self.gallery_dir, relabel=True
        )

        super().__init__(
            dataset_dir=self._dataset_path,
            train_data=train_data,
            gallery_data=gallery_data,
            query_data=query_data,
            domain=self._domain,
        )

        if verbose:
            print(f"=> {self._domain} loaded")
            self.show_dataset_info()

        (
            self.num_train_imgs,
            self.num_train_aids,
            self.num_train_cams,
            self.num_train_views,
            _,
            _,
        ) = get_dataset_info(self.train_data)
        (
            self.num_gallery_imgs,
            self.num_gallery_aids,
            self.num_gallery_cams,
            self.num_gallery_views,
            _,
            _,
        ) = get_dataset_info(self.gallery_data)
        (
            self.num_query_imgs,
            self.num_query_aids,
            self.num_query_cams,
            self.num_query_views,
            _,
            _,
        ) = get_dataset_info(self.query_data)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self._dataset_path):
            raise RuntimeError("'{}' is not available".format(self._dataset_path))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))

    def read_test_data(self, query_dir: str, gallery_dir: str, relabel: bool = True):
        def _load_data_from_directory(dir_path):
            files_list = listdir_nonhidden(path=dir_path)
            image_paths = []
            for file in files_list:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(dir_path, file))
            return image_paths

        def _parse_name(img_path):
            image_name, _ = os.path.splitext(os.path.basename(img_path))
            comps = image_name.split("_")
            if len(comps) < 2:
                raise ValueError(
                    f"Filename '{image_name}' missing 'aid_camid_...' parts."
                )
            aid = comps[0]
            camid = comps[1]
            return aid, camid

        def _build_mapping(query_paths, gallery_paths):
            mapping = {}
            next_id = 0
            for paths in (query_paths, gallery_paths):
                for p in paths:
                    aid, _ = _parse_name(p)
                    if aid not in mapping:
                        print(f"{aid}: {next_id}")
                        mapping[aid] = next_id
                        next_id += 1
            return mapping

        def _build_datums(image_paths, relabel, aid2label):
            datums = []
            for img_p in image_paths:
                aid_str, camid = _parse_name(img_p)
                aid_int = aid2label[aid_str] if relabel else int(aid_str)
                datums.append(
                    Datum(
                        img_path=img_p,
                        aid=aid_int,
                        camid=camid,
                        viewid=-1,
                        domain_label=self.domain_label,
                    )
                )
            return datums

        query_paths = _load_data_from_directory(query_dir)
        gallery_paths = _load_data_from_directory(gallery_dir)

        aid2label = _build_mapping(query_paths, gallery_paths) if relabel else {}

        query_data = _build_datums(query_paths, relabel=relabel, aid2label=aid2label)
        gallery_data = _build_datums(
            gallery_paths, relabel=relabel, aid2label=aid2label
        )

        return query_data, gallery_data, aid2label

    def read_data(self, data_dir, relabel=False):
        def _load_data_from_directory(dir_path):
            files_list = listdir_nonhidden(path=dir_path)
            image_paths = []
            for file in files_list:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_p = os.path.join(dir_path, file)
                    image_paths.append(img_p)
            return image_paths

        data_paths = _load_data_from_directory(data_dir)
        label = 0
        aid2label = dict()
        aid_container, camid_container = set(), set()
        img_datums = []

        for img_p in data_paths:
            image_name, ext = os.path.splitext(os.path.basename(img_p))
            components_list = image_name.split("_")
            # print(components_list)
            aid, camid = int(components_list[0]), components_list[1]
            aid_container.add(aid)
            camid_container.add(camid)
            if aid not in aid2label:
                aid2label[aid] = label
                label += 1

            if relabel:
                aid = aid2label[aid]

            img_datum = Datum(
                img_path=img_p,
                aid=aid,
                camid=camid,
                viewid=-1,
                domain_label=self.domain_label,  # Store domain label based on MultiReID assignment
            )
            img_datums.append(img_datum)

        return img_datums
