import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    # dataset 의 전 처리를 담당 한다.
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    # dataset 의 길이, 즉 총 샘플의 수를 적어 준다.
    def __len__(self):
        return len(self.x_data)

    # dataset 에서 특정 1개의 sample 을 가져 온다.
    def __getitem__(self, item):
        x = torch.FloatTensor(self.x_data[item])
        y = torch.FloatTensor(self.y_data[item])

        return x, y
