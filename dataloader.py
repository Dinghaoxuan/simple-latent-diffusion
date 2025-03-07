import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from ldm.data import Txt2ImgIterableBaseDataset
import torch
import numpy as np
from functools import partial
from ldm.util import instantiate_from_config


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    
    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size : (worker_id + 1) * split_size] # 不同的worker加载不同部分的数据
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id) # 设定不同的worker的随机种子
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)
        
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModeuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None, wrap=False, 
                 num_workers=None, use_worker_init_fn=False, shuffle_test_loader=False, shuffle_val_dataloader=False):
        super().__init__()
        
        self.batch_size = batch_size
        self.dataset_config = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        
        if train is not None:
            self.dataset_config['train'] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_config['validation'] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_config['test'] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_config['predict'] = predict
            self.predict_dataloader = self._predict_dataloader
        
        self.wrap = wrap
            
    def prepare_data(self):
        for data_cfg in self.dataset_config.values():
            instantiate_from_config(data_cfg)
            
    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_config[k])) for k in self.dataset_config.keys()
        )
        if self.wrap: # 转换为torch.utils.data.Dataset
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

            
    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.dataset_config['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)
        
        
    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        
        return DataLoader(self.datasets["validation"], 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          worker_init_fn=init_fn)
    
    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)
        
        return DataLoader(self.datasets["test"], 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=shuffle,
                          worker_init_fn=init_fn)
    
    def _predict_dataloader(self):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          worker_init_fn=init_fn)
        
        
        
        
        
        
        
        
        
        
