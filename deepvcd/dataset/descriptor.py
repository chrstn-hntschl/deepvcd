import warnings
import os
import pathlib
import numpy as np
import random
import logging
from abc import ABC, abstractmethod

import yaml
from tqdm import tqdm
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
    warnings.warn("LibYAML not found - yaml descriptor will be loaded using the much slower PyYAML package.")

import tensorflow as tf

log = logging.getLogger(__name__)

class DatasetDescriptor(object):
    def __init__(self, name, version, basepath='/'):
        """
        Create a dataset object to hold and manage dataset metadata (name, version) and groundtruth annotation data.

        :param name:
        :param version:
        :param basepath:
        """
        self.basepath = basepath
        self.categories = []
        self.neg_category = None
        self.name = name
        self.version = version

        self.ground_truth = {
            'train': {
                'gt': {},
                'images': {},
            },
            'val': {
                'gt': {},
                'images': {},
            },
            'test': {
                'gt': {},
                'images': {},
            },
        }

    def add_labeled_images(self, subset, category, fnames):
        if category not in self.categories:
            self.categories.append(category)

        self.ground_truth[subset]['gt'].setdefault(category, [])
        for fname in fnames:
            h = hash(fname)
            for s in {'train', 'val', 'test'} - {subset}:
                assert h not in self.ground_truth[s]['images'], \
                        "image '{fname}' found in '{s}' set - '{subset}' set and '{s}' set need to be disjoint!".format(
                        fname=fname, s=s, subset=subset)
            if not os.path.exists(os.path.join(self.basepath, fname)):
                warnings.warn("No such image file: '{fname}'".format(fname=os.path.join(self.basepath, fname)))
            self.ground_truth[subset]['images'][h] = fname
            self.ground_truth[subset]['gt'][category].append(h)

    def set_negative_category(self, category):
        assert category in self.categories, "Category not found in dataset!"
        self.neg_category = category

    @property
    def dataset_name(self):
        return self.dataset_name

    def get_labels(self):
        # FIXME: pyvcd.core.VCDataset uses std::set<std::string> for internal storage of category labels
        #        This is wrong - we should keep the order as given in the descriptor yaml!
        #        As long as this is not fixed, we sort the labels in pure python as well to copy this
        #        behavior
        return sorted(self.categories)

    def _get_subset_images(self, subset, category=None, pos_only=False):
        if category is None:
            return [os.path.join(self.basepath, fname) for fname in self.ground_truth[subset]['images'].values()]
        else:
            posids = self.ground_truth[subset]['gt'][category]

            pos = [os.path.join(self.basepath, self.ground_truth[subset]['images'][id]) for id in posids]

            if pos_only:
                return pos
            else:
                if self.neg_category is not None:
                    negids = self.ground_truth[subset]['gt'][self.neg_category]
                else:
                    negids = list(set(self.ground_truth[subset]['images'].keys()) - set(posids))

                neg = [os.path.join(self.basepath, self.ground_truth[subset]['images'][id]) for id in negids]

                return pos, neg

    def get_train_images(self, category=None, pos_only=False):
        return self._get_subset_images(subset='train', category=category, pos_only=pos_only)

    def get_val_images(self, category=None, pos_only=False):
        return self._get_subset_images(subset='val', category=category, pos_only=pos_only)

    def get_test_images(self, category=None, pos_only=False):
        return self._get_subset_images(subset='test', category=category, pos_only=pos_only)

    @property
    def dataset_version(self):
        return self.version
    
    def get_tfdataset(self, subset: str, shuffle_files: bool=False, seed: int=None):
        label_indices = dict()
        img_file_paths = list()
        img_labels = list()
        for label_idx,label in enumerate(sorted(self.get_labels())):
            label_indices[label_idx] = label
            imgs = self._get_subset_images(subset=subset, category=label, pos_only=True)
            img_file_paths.extend(imgs)
            img_labels.extend(len(imgs)*[label_idx])

        if shuffle_files:
            if seed is None:
                seed_ = 0
            else:
                seed_ = seed
            combined = list(zip(img_file_paths, img_labels))
            random.Random(seed_).shuffle(combined)
            img_file_paths[:], img_labels[:] = zip(*combined)

        img_file_paths = tf.constant(img_file_paths)
        img_labels = tf.constant(img_labels)

        return tf.data.Dataset.from_tensor_slices((img_file_paths, img_labels)), label_indices


def get_cross_val_folds(ds_descriptor, n_folds=4, seed=None):
    """
    Splits the train set of the given DatasetDescriptor into `n_folds` cross validation folds.
    Returns n_folds DatasetDescriptors with train and val subsets set according to generated folds.
    If seed is not None, it will be used to make the random splits reproducible.
    """
    X = ds_descriptor.get_train_images()
    index = dict([(hash(i), idx) for idx,i in enumerate(X)])
    categories = ds_descriptor.get_labels()
    y = np.zeros( (len(X), len(categories)) )

    for c_idx, category in enumerate(categories):
        pos_imgs = ds_descriptor.get_train_images(category=category, pos_only=True)
        for pos in  pos_imgs:
            assert y[index[hash(pos)], c_idx] == 0, "Image-label assignment redundant"
            y[index[hash(pos)], c_idx] = 1
    
    multi_label = np.any(np.sum(y, axis=1) > 1)         
    
    if not multi_label:
        # single-label:
        from sklearn.model_selection import StratifiedKFold
        
        y = np.where(y==1)[1]
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_descrs = list()
    
        for fold, (train, val) in enumerate(skf.split(X=X, y=y)):
            fold_descr = DatasetDescriptor(name=ds_descriptor.name+"_fold{fold}".format(fold=fold),
                                           version=ds_descriptor.version, basepath='')
            for train_idx in train:
                fold_descr.add_labeled_images(subset='train', category=y[train_idx], fnames=[X[train_idx]])
            for val_idx in val:
                fold_descr.add_labeled_images(subset='val', category=y[val_idx], fnames=[X[val_idx]])
            fold_descrs.append(fold_descr)

    else:
        # multi-label:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
                
        mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_descrs = list()
        for fold, (train, val) in enumerate(mskf.split(X=X, y=y)):
            fold_descr = DatasetDescriptor(name=ds_descriptor.name+"_fold{fold}".format(fold=fold),
                                           version=ds_descriptor.version, basepath='')
        
            for c_idx in range(len(categories)):
                pos_train_idx = set(train).intersection(set(np.where(y[:,c_idx]==1)[0]))
                pos_val_idx = set(val).intersection(set(np.where(y[:,c_idx]==1)[0]))
                
                fold_descr.add_labeled_images(subset='train', category=categories[c_idx], fnames=list(np.asarray(X)[list(pos_train_idx)]))
                fold_descr.add_labeled_images(subset='val', category=categories[c_idx], fnames=list(np.asarray(X)[list(pos_val_idx)]))

            fold_descrs.append(fold_descr)

    return fold_descrs


class DescriptorLoader(ABC):
    @abstractmethod
    def get_dataset(self) -> DatasetDescriptor:
        pass

class YAMLLoader(DescriptorLoader):
    def __init__(self, yaml_file, basepath=None) -> None:
        """
        Loads a DatasetDescriptor object from a yaml file. Use the static `read` method for convenience.
        :param yaml_file: the yaml serialization to load the descriptor from.
        :param basepath: if the actual basepath differs from the basepath in the descriptor file, use this to rebase.
        """
        self.yaml_file = yaml_file
        self.basepath = basepath

    def get_dataset(self) -> DatasetDescriptor:
        data = yaml.load(open(self.yaml_file, 'r'), Loader=Loader)
        if self.basepath is not None:
            basepath = self.basepath
        else:
            if 'basepath' in data:
                if os.path.isabs(data['basepath']):
                    basepath = data['basepath']
                else:
                    basepath = os.path.join(os.path.dirname(self.yaml_file), data['basepath'])
            else:
                basepath = '/'

        dataset = DatasetDescriptor(name=data['dataset'], version=data.get('version', 'undefined'), basepath=basepath)
        subsets = ['train', 'val', 'test']
        for subset in subsets:
            if subset in data:
                for category in data['categories']:
                    try:
                        fnames = [data[subset]['images'][id] for id in data[subset]['gt'][category]]
                    except KeyError:
                        fnames = [data[subset]['images'][str(id)] for id in data[subset]['gt'][category]]
                    dataset.add_labeled_images(subset=subset, category=category, fnames=fnames)
        if 'neg_category' in data:
            dataset.set_negative_category(data['neg_category'])

        return dataset

    @staticmethod
    def read(yaml_file, basepath=None):
        return YAMLLoader(yaml_file, basepath=basepath).get_dataset()


class DirectoryLoader(DescriptorLoader):
    def __init__(self, dataset_path:str) -> None:
        self.dataset_dir = pathlib.Path(dataset_path)
        if not self.dataset_dir.is_dir():
            raise ValueError("Dataset path #{0}' is not a valid path!".format(dataset_path))

    def get_dataset(self) -> DatasetDescriptor:
        name = self.dataset_dir.name
        version = "undefined"
        dataset = DatasetDescriptor(name=name, version=version, basepath=str(self.dataset_dir))

        subsets = ["train", "val", "test"]
        for subset in subsets:
            subset_dir = self.dataset_dir / subset
            if subset_dir.is_dir():
                log.info(f"Loading subset '{subset}'")
                categories = list()
                for class_dir in subset_dir.iterdir():
                    if class_dir.is_dir():
                        categories.append(class_dir.name)

                for category in tqdm(categories):
                    fnames = [p.resolve() for p in pathlib.Path(class_dir).glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                    dataset.add_labeled_images(subset=subset, category=category, fnames=fnames)

        return dataset

    @staticmethod
    def load(dataset_dir):
        return DirectoryLoader(dataset_dir).get_dataset()