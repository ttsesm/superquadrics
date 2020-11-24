import datetime
import time
import pickle

import torch
from tqdm import tqdm
import sys
sys.path.append("/home/niko/workspace/torch-points3d/torch_points3d/datasets/utils/")
import random
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from torch_points3d.metrics.classification_tracker import ClassificationTracker
from torch_points3d.metrics.regression_tracker import RegressionTracker
from superquadric_generator import Superquadric
from torch_points3d.datasets.base_dataset import BaseDataset

from torch_points3d.applications.rsconv import RSConv

sq_offset_range = (25, 230)
sq_dimensions_range = (25, 75)
epsilon_range = (0.1, 1.0)
dimension_max = 305

NUM_WORKERS = 0
BATCH_SIZE = 12

from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq

def train_epoch(device):
    model.to(device)
    model.train()
    tracker.reset("train")
    train_loader = dataset.train_dataloader
    iter_data_time = time.time()
    with Ctq(train_loader) as tq_train_loader:
        for i, data in enumerate(tq_train_loader):
            t_data = time.time() - iter_data_time
            iter_start_time = time.time()
            optimizer.zero_grad()
            data.to(device)
            model.forward(data)
            model.backward()
            optimizer.step()
            if i % 10 == 0:
                tracker.track(model)

            tq_train_loader.set_postfix(
                **tracker.get_metrics(),
                data_loading=float(t_data),
                iteration=float(time.time() - iter_start_time),
            )
            iter_data_time = time.time()


def test_epoch(device):
    model.to(device)
    model.eval()
    tracker.reset("test")
    test_loader = dataset.test_dataloaders[0]
    iter_data_time = time.time()
    with Ctq(test_loader) as tq_test_loader:
        for i, data in enumerate(tq_test_loader):
            t_data = time.time() - iter_data_time
            iter_start_time = time.time()
            data.to(device)
            model.forward(data)
            tracker.track(model)

            tq_test_loader.set_postfix(
                **tracker.get_metrics(),
                data_loading=float(t_data),
                iteration=float(time.time() - iter_start_time),
            )
            iter_data_time = time.time()

class RSConvCLassifier(torch.nn.Module):
    def __init__(self, USE_NORMAL):
        super().__init__()
        self.encoder = RSConv("encoder", input_nc=3 * USE_NORMAL, output_nc=25, num_layers=4)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    @property
    def conv_type(self):
        """ This is needed by the dataset to infer which batch collate should be used"""
        return self.encoder.conv_type

    def get_output(self):
        """ This is needed by the tracker to get access to the ouputs of the network"""
        return self.output

    def get_labels(self):
        """ Needed by the tracker in order to access ground truth labels"""
        return self.labels

    def get_current_losses(self):
        """ Entry point for the tracker to grab the loss """
        return {"loss_class": float(self.loss_class)}

    def forward(self, data):
        # Set labels for the tracker
        self.labels = data.y.squeeze()

        # Forward through the network
        data_out = self.encoder(data)
        self.output = self.log_softmax(data_out.x.squeeze())

        # Set loss for the backward pass
        self.loss_class = torch.nn.functional.nll_loss(self.output, self.labels)

    def backward(self):
        self.loss_class.backward()


class RSConvRegressor(torch.nn.Module):
    def __init__(self, USE_NORMAL):
        super().__init__()
        self.encoder = RSConv("encoder", input_nc=3 * USE_NORMAL, output_nc=8, num_layers=4)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.loss_function = torch.nn.MSELoss()

    @property
    def conv_type(self):
        """ This is needed by the dataset to infer which batch collate should be used"""
        return self.encoder.conv_type

    def get_output(self):
        """ This is needed by the tracker to get access to the ouputs of the network"""
        return self.output

    def get_labels(self):
        """ Needed by the tracker in order to access ground truth labels"""
        return self.labels

    def get_current_losses(self):
        """ Entry point for the tracker to grab the loss """
        return {"loss": float(self.loss)}

    def forward(self, data):
        # Set labels for the tracker
        self.labels = data.y.squeeze()

        # Forward through the network
        data_out = self.encoder(data)
        self.output = data_out.x.squeeze()

        # Set loss for the backward pass
        # self.loss_class = torch.nn.functional.MSE(self.output, self.labels)
        self.loss = self.loss_function(self.labels, self.output)

    def backward(self):
        self.loss.backward()


def get_random_superquadric_parameters():
    x0, y0, z0 = random.sample(range(sq_offset_range[0], sq_offset_range[1]), 3)
    a1, a2, a3 = random.sample(range(sq_dimensions_range[0], sq_dimensions_range[1]), 3)
    epsilon1, epsilon2 = list(np.random.uniform(low=epsilon_range[0], high=epsilon_range[1], size=(2,)))
    return a1, a2, a3, epsilon1, epsilon2, x0, y0, z0


class SuperQuadricsClassification(Dataset):
    def __init__(self, dataset_size, points_count, dimension_max, do_normalize=True, transform=None,
                 classification=True):
        """
        Args:
            number of superquadrics to be gnerated.
        """
        super().__init__()

        used_combinations = []
        self.pos = np.zeros((dataset_size, points_count, 3))
        self.norm = np.zeros((dataset_size, points_count, 3))
        if classification:
            self.y = np.zeros((dataset_size, 3))
        else:
            self.y = np.zeros((dataset_size, 8))
        pbar = tqdm(total=dataset_size)
        idx = 0
        while idx < dataset_size:
            combination_is_valid = False
            while not combination_is_valid:
                a1, a2, a3, e1, e2, x0, y0, z0 = get_random_superquadric_parameters()
                current_parameters_str = '{}#{}#{}#{}#{}#{}#{}#{}'.format(a1, a2, a3, e1, e2, x0, y0, z0)
                if current_parameters_str not in used_combinations:
                    combination_is_valid = True
            used_combinations.append(current_parameters_str)
            superquadric = Superquadric(a1, a2, a3, e1, e2, x0, y0, z0, dimension_max)
            try:
                points, normals = superquadric.get_grid(points_count)
                if do_normalize:
                    max_offset = sq_dimensions_range[1] + sq_offset_range[1]
                    points = list(map(lambda x: [item / max_offset for item in x], points))

            except:
                # logger.warning(
                #     'Unable to generate superquadric with following parameters: a1 = {}, a2 = {}, a3 = {}, e1 = {}, e2 = {}, x0 = {}, y0 = {}, z0 = {}'.format(
                #         a1, a2, a3, e1, e2, x0, y0, z0))
                continue
            self.pos[idx] = np.array(points)
            self.norm[idx] = normals
            if classification:
                boundaries = np.linspace(0, 1, num=6)
                class_label = self.get_class_num(boundaries, e1, e2)
                self.y[idx] = np.array([class_label, e1, e2])
            else:
                self.y[idx] = np.array([a1, a2, a3, e1, e2, x0, y0, z0])
            idx += 1
            pbar.update(1)
        pbar.close()

    def get_interval_num(self, boundaries, epsilon):
        boundaries_intervals = [(boundaries[i], boundaries[i+1]) for i in range(boundaries.size - 1)]
        for i, (val_min, val_max) in enumerate(boundaries_intervals):
            if epsilon >= val_min and epsilon < val_max:
                return i

    def get_class_num(self, boundaries, epsilon1, epsilon2):
        interval1 = self.get_interval_num(boundaries, epsilon1)
        interval2 = self.get_interval_num(boundaries, epsilon2)
        return float(5 * interval1 + interval2)

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        norm = torch.from_numpy(self.norm[idx]).float()
        pos = torch.from_numpy(self.pos[idx]).float()
        y = torch.from_numpy(self.y[idx])[0].long()
        return Data(norm=norm, x=norm, pos=pos, y=y)

class SuperQuadricsRegressionShape(Dataset):
    def __init__(self, dataset_size, points_count, dimension_max, do_normalize=True, transform=None):
        """
        Args:
            number of superquadrics to be gnerated.
        """
        super().__init__()

        used_combinations = []
        self.pos = np.zeros((dataset_size, points_count, 3))
        self.norm = np.zeros((dataset_size, points_count, 3))
        self.y = np.zeros((dataset_size, 8))
        pbar = tqdm(total=dataset_size)
        idx = 0
        while idx < dataset_size:
            combination_is_valid = False
            while not combination_is_valid:
                a1, a2, a3, e1, e2, x0, y0, z0 = get_random_superquadric_parameters()
                current_parameters_str = '{}#{}#{}#{}#{}#{}#{}#{}'.format(a1, a2, a3, e1, e2, x0, y0, z0)
                if current_parameters_str not in used_combinations:
                    combination_is_valid = True
            used_combinations.append(current_parameters_str)
            superquadric = Superquadric(a1, a2, a3, e1, e2, x0, y0, z0, dimension_max)
            try:
                points, normals = superquadric.get_grid(points_count)
                if do_normalize:
                    max_offset = sq_dimensions_range[1] + sq_offset_range[1]
                    points = list(map(lambda x: [item / max_offset for item in x], points))

            except:
                # logger.warning(
                #     'Unable to generate superquadric with following parameters: a1 = {}, a2 = {}, a3 = {}, e1 = {}, e2 = {}, x0 = {}, y0 = {}, z0 = {}'.format(
                #         a1, a2, a3, e1, e2, x0, y0, z0))
                continue
            self.pos[idx] = np.array(points)
            self.norm[idx] = normals
            self.y[idx] = np.array([a1, a2, a3, e1, e2, x0, y0, z0])
            idx += 1
            pbar.update(1)
        pbar.close()

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        norm = torch.from_numpy(self.norm[idx]).float()
        pos = torch.from_numpy(self.pos[idx]).float()
        y = torch.from_numpy(self.y[idx]).double()
        return Data(norm=norm, x=norm, pos=pos, y=y)


class SuperQuadricsClassificationDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        # number = dataset_opt.number
        # if str(number) not in self.AVAILABLE_NUMBERS:
        #     raise Exception("Only ModelNet10 and ModelNet40 are available")
        self.train_dataset = SuperQuadricsClassification(dataset_size=20000, points_count=2048, dimension_max=305, transform=self.train_transform)
        self.test_dataset = SuperQuadricsClassification(dataset_size=5000, points_count=2048, dimension_max=305, transform=self.test_transform)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return ClassificationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


class SuperQuadricsRegressionShapeDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.train_dataset = SuperQuadricsRegressionShape(dataset_size=40000, points_count=2048, dimension_max=305, transform=self.train_transform)
        self.test_dataset = SuperQuadricsRegressionShape(dataset_size=5000, points_count=2048, dimension_max=305, transform=self.test_transform)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return RegressionTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


if __name__ == '__main__':
    USE_NORMAL = True  # @param {type:"boolean"}
    DIR = ""
    yaml_config = """
    task: classification
    class: modelnet.ModelNetDataset
    name: modelnet
    dataroot: %s
    pre_transforms:
        - transform: NormalizeScale
        - transform: GridSampling3D
          lparams: [0.02]
    train_transforms:
        - transform: FixedPoints
          lparams: [2048]
        - transform: RandomNoise
        - transform: RandomRotate
          params:
            degrees: 180
            axis: 2
        - transform: AddFeatsByKeys
          params:
            feat_names: [norm]
            list_add_to_x: [%r]
            delete_feats: [True]
    test_transforms:
        - transform: FixedPoints
          lparams: [2048]
        - transform: AddFeatsByKeys
          params:
            feat_names: [norm]
            list_add_to_x: [%r]
            delete_feats: [True]
    """ % (os.path.join(DIR, "data"), USE_NORMAL, USE_NORMAL)

    from omegaconf import OmegaConf
    params = OmegaConf.create(yaml_config)
    task = 'regression'
    GENERATE_DATASET = False
    if GENERATE_DATASET:
        if task == 'classification':
            dataset = SuperQuadricsClassificationDataset(params)
        elif task == 'regression':
            dataset = SuperQuadricsRegressionShapeDataset(params)
        else:
            raise Exception(f'Unknown task {task}')
        with open(f'/home/niko/workspace/torch-points3d/torch_points3d/data/sq_dataset_{task}.pkl', 'wb') as output:
            pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'/home/niko/workspace/torch-points3d/torch_points3d/data/sq_dataset_{task}.pkl', 'rb') as input:
            dataset = pickle.load(input)
    print(dataset)
    # Setup the data loaders
    if task == 'regression':
        model = RSConvRegressor(USE_NORMAL)
    else:
        model = RSConvCLassifier(USE_NORMAL)
    dataset.create_dataloaders(
        model,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        precompute_multi_scale=False
    )
    print(next(iter(dataset.test_dataloaders[0])))
    # Setup the tracker and actiavte tensorboard loging
    logdir = "/home/niko/workspace/torch-points3d/runs"
    logdir = os.path.join(logdir, str(datetime.datetime.now()))
    os.mkdir(logdir)
    os.chdir(logdir)
    tracker = dataset.get_tracker(False, True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    EPOCHS = 200
    for i in range(EPOCHS):
        print("=========== EPOCH %i ===========" % i)
        time.sleep(0.5)
        train_epoch('cuda')
        tracker.publish(i)
        test_epoch('cuda')
        tracker.publish(i)
        if i % 10 == 0:
            torch.save(model.state_dict(), f'/home/niko/workspace/torch-points3d/saved_models/classification_{i}_dict_model.pt')