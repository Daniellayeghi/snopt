import numpy as np
import torch

from torchvision import transforms
from prefetch_generator import BackgroundGenerator
import util

import torchvision.datasets as torch_data


def _gen_mini_dataset(dataset, dataset_ratio):
    n_dataset = dataset.shape[0]
    n_mini_dataset = int(dataset_ratio*n_dataset)
    s = torch.from_numpy(np.random.choice(
        np.arange(n_dataset, dtype=np.int64), n_mini_dataset, replace=False)
    )
    return dataset[s]


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class TabularLoader:
    def __init__(self,opt, data, batch_size=None, shuffle=True):

        self.data_size = data.shape[0]
        self.opt = opt
        self.device = opt.device

        self.data = data.to(opt.device)
        self.batch_size = opt.batch_size if batch_size is None else batch_size
        self.shuffle = shuffle

        self.input_dim = data.shape[-1]
        self.output_dim= [data.shape[-1]]

        loc = torch.zeros(data.shape[-1]).to(opt.device)
        covariance_matrix = torch.eye(data.shape[-1]).to(opt.device) # TODO(Guan) scale down the cov ?
        self.p_z0 = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
        self._reset_idxs()
        self.data_size = len(self.idxs_by_batch_size)

    def _reset_idxs(self):
        idxs = torch.randperm(self.data.shape[0]) if self.shuffle else torch.arange(self.data.shape[0])
        self.idxs_by_batch_size = idxs.split(self.batch_size)
        self.batch_idx = 0

    def __len__(self):
        return self.data_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx >= len(self.idxs_by_batch_size):
            self._reset_idxs()
            raise StopIteration

        s = self.idxs_by_batch_size[self.batch_idx]
        self.batch_idx += 1
        x = self.data[s]
        logp_diff_t1 = torch.zeros(x.shape[0], 1, device=x.device)
        return (x, logp_diff_t1), self.p_z0



def get_img_loader(opt, test_batch_size=1000):
    print(util.magenta("loading image data..."))

    dataset_builder, root, input_dim, output_dim = {
        'mnist':   [torch_data.MNIST,  'data/img/mnist',  [1,28,28], 10],
        'SVHN':    [torch_data.SVHN,   'data/img/svhn',   [3,32,32], 10],
        'cifar10': [torch_data.CIFAR10,'data/img/cifar10',[3,32,32], 10],
    }.get(opt.problem)
    opt.input_dim = input_dim
    opt.output_dim = output_dim

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    feed_dict = dict(download=True, root=root, transform=transform)
    train_dataset = dataset_builder(**feed_dict) if opt.problem=='SVHN' else dataset_builder(train=True, **feed_dict)
    test_dataset  = dataset_builder(**feed_dict) if opt.problem=='SVHN' else dataset_builder(train=False, **feed_dict)

    feed_dict = dict(num_workers=2, drop_last=True)
    train_loader = DataLoaderX(train_dataset, batch_size=opt.batch_size, shuffle=True, **feed_dict)
    test_loader  = DataLoaderX(test_dataset, batch_size=test_batch_size, shuffle=False, **feed_dict)

    return train_loader, test_loader
