"""
Main file for the project.
"""

import numpy as np
import math
import random
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from path import Path
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

random.seed = 42
path = Path("datasets/ModelNet10")

# class for loading the data
folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
classes = {folder: i for i, folder in enumerate(folders)};


def read_off(file):
    """
    Reads a .off file and returns a tuple of the vertices and faces.
        Parameters
        ----------
            file : str
                path to the .off file

        Returns
        -------
            vertices, faces
    """
    if 'OFF' != file.readline().strip():
        raise ('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


with open(path / "bed/train/bed_0001.off", 'r') as f:
    verts, faces = read_off(f)

i, j, k = np.array(faces).T
x, y, z = np.array(verts).T


class PointSampler(object):
    """
    Samples given number of points from a mesh.
    """

    def __init__(self, output_size):
        """
        Parameters
        ----------
            output_size: int
                number of points to sample
        """
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        """
        Input mest to sample from.
            Parameters
            ----------
                mesh: (vertices, faces) tuple

            Returns
            -------
                sampled_mesh: samples points from the mesh
        """

        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                        weights=areas,
                                        cum_weights=None,
                                        k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points


pointcloud = PointSampler(3000)((verts, faces))


class Normalize(object):
    """
    Normalizes the pointcloud to have mean 0
    """

    def __call__(self, pointcloud):
        """
        Applies the normalization to the pointcloud.
            Parameters
            ----------
                pointcloud: list
                    pointcloud data to normalize

            Returns
            -------
                normalized pointcloud data
        """
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud


norm_pointcloud = Normalize()(pointcloud)


class RandRotation_z(object):
    """
    Randomly rotates the pointcloud around the z axis.
    """

    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                               [math.sin(theta), math.cos(theta), 0],
                               [0, 0, 1]])

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud


class RandomNoise(object):
    """
    Adds random noise to the pointcloud.
    """

    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


rot_pointcloud = RandRotation_z()(norm_pointcloud)
noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)


class ToTensor(object):
    """
    Converts the pointcloud to a tensor.
    """

    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)


ToTensor()(noisy_rot_pointcloud)


def default_transforms():
    """
    Returns a list of default transformations.
        Returns
        -------
            list of transformations
    """
    return transforms.Compose([
        PointSampler(1024),
        Normalize(),
        ToTensor()
    ])


class PointCloudData(Dataset):
    """
    Custom dataset implementation for Pointcloud dataset.
    """

    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        """
        Parameters
        ----------
            root_dir: str
                root directory of the dataset
            valid: bool
                whether to use the validation set
            folder: str
                folder to use
            transform: list
                list of transformations to apply to the dataset
        """

        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir / dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir / Path(category) / folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir / file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        """
        Returns the length of the dataset.
            Returns
            -------
                length of the dataset
        """
        return len(self.files)

    def __preproc__(self, file):
        """
        Preprocesses the pointcloud.
            Parameters
            ----------
                file: str
                    sample pointcloud file to preprocess

            Returns
            -------
                preprocessed pointcloud data
        """

        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        """
        Returns the pointcloud at the given index.
            Parameters
            ----------
                idx: int
                    index of the pointcloud to return

            Returns
            -------
                (pointcloud data, category)
        """
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud,
                'category': self.classes[category]}


# defining the augmentation (transformation) pipeline
train_transforms = transforms.Compose([
    PointSampler(1024),
    Normalize(),
    RandRotation_z(),
    RandomNoise(),
    ToTensor()
])

# defining the dataset
train_ds = PointCloudData(path, transform=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)

# dictionary of the dataset class indexes
inv_classes = {i: cat for cat, i in train_ds.classes.items()};

# instantiating the dataloader
print('Train dataset size: ', len(train_ds), flush=True)
print('Valid dataset size: ', len(valid_ds), flush=True)
print('Number of classes: ', len(train_ds.classes), flush=True)
print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size(), flush=True)
print('Class: ', inv_classes[train_ds[0]['category']], flush=True)

# Torch dataloader wrapper  for the datasets
train_loader = DataLoader(dataset=train_ds, batch_size=16, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=64)


# defining the Tnet module of the model
class Tnet(nn.Module):
    def __init__(self, k=3):
        """
        Tnet module of the model.
        Parameters
        ----------
        k : int
            return dimension of the module
        """
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        """
        Forward pass of the model.
        Parameters
        ----------
        input: torch.Tensor
            input pointcloud data

        Returns
        -------
            output: torch.Tensor of shape (batch_size, k, k)
        """
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        # initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if xb.is_cuda:
            init = init.cuda()
        matrix = self.fc3(xb).view(-1, self.k, self.k) + init
        return matrix


class Transform(nn.Module):
    """
    Transform module of the model.
    """
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)
        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64


class PointNet(nn.Module):
    """
    PointNet model.
    """
    def __init__(self, classes=10):
        """
        Parameters
        ----------
        classes: int
            number of classes   (default: 10)
        """
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        """
        Forward pass of the model.
        Parameters
        ----------
        input: int
            input pointcloud data

        Returns
        -------
            output: torch.Tensor of shape (batch_size, number_of_classes)
        """
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64


def pointnetloss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    """
    Loss function of the model.
    Parameters
    ----------
    outputs: torch.Tensor
        output of the model
    labels: torch.Tensor
        ground truth labels
    m3x3: torch.Tensor
        3x3 intermediate matrix
    m64x64: torch.Tensor
        64x64 intermediate matrix
    alpha: float
        weight of second loss term (default: 0.0001)

    Returns
    -------

    """
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)


# selecting the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, flush=True)

# defining the network
pointnet = PointNet()
pointnet.to(device);

# defining the optimizer
optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)


def train(pointnet, train_loader, val_loader=None, epochs=15, save=True):
    """
    Training loop.
    Parameters
    ----------
    pointnet: object
        pointnet model
    train_loader: object
        training data loader
    val_loader: object
        validation data loader
    epochs: int
        number of epochs (default: 15)
    save: bool
        whether to save the model (default: True)

    Returns
    -------
        None

    """
    for epoch in range(epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                      (epoch + 1, i + 1, len(train_loader), running_loss / 10), flush=True)
                running_loss = 0.0

        pointnet.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1, 2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %.2f %%' % val_acc, flush=True)

        # save the model
        if save:
            torch.save(pointnet.state_dict(), "save_" + str(epoch) + ".pth")


# run the training loop
train(pointnet, train_loader, valid_loader, save=True)
