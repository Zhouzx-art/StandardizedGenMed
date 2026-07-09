from torch.utils.data import DataLoader
from dataset import LIDCDataset, LIDCInDataset
from dataset import EMIDECDataset, EMIDECInDataset
from dataset import ISICDataset,ISICInDataset
from dataset import FUNDUSDataset,FUNDUSInDataset

def get_inference_dataloader(dataset_root_dir, test_txt_dir,batch_size=1, drop_last=False, data_type=''):
    if data_type == 'lidc':
        train_dataset = LIDCInDataset(root_dir=dataset_root_dir, test_txt_dir=test_txt_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )
    elif data_type == 'isic'or data_type == 'polyp' or data_type == 'prostate' or data_type == 'nuclei' or data_type == 'fets' or data_type == 'chaos':
        train_dataset = ISICInDataset(root_dir=dataset_root_dir, test_txt_dir=test_txt_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )
    elif data_type == 'fundus':
        train_dataset = FUNDUSInDataset(root_dir=dataset_root_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )
    elif data_type == 'emidec':
        train_dataset = EMIDECInDataset(root_dir=dataset_root_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )
    return loader

def get_train_dataset(cfg):
    if cfg.dataset.data_type == 'lidc':
        train_dataset = LIDCDataset(root_dir=cfg.dataset.root_dir, test_txt_dir=cfg.dataset.test_txt_dir)
        sampler = None
    elif cfg.dataset.data_type == 'isic'or cfg.dataset.data_type == 'polyp' or cfg.dataset.data_type == 'prostate' or cfg.dataset.data_type == 'nuclei' or cfg.dataset.data_type == 'fets' or cfg.dataset.data_type == 'chaos':
        train_dataset = ISICDataset(root_dir=cfg.dataset.root_dir, test_txt_dir=cfg.dataset.test_txt_dir)
        sampler = None
    elif cfg.dataset.data_type == 'fundus':
        train_dataset = FUNDUSDataset(root_dir=cfg.dataset.root_dir)
        sampler = None
    elif cfg.dataset.data_type == 'emidec':
        train_dataset = EMIDECDataset(root_dir=cfg.dataset.root_dir)
        sampler = None
    return train_dataset, sampler


