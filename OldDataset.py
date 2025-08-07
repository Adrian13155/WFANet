import torch.utils.data as data
import numpy as np
import torch

class Gaussian_dataset(data.Dataset):
    def __init__(self, mat_data):
        super(Gaussian_dataset, self).__init__()

        gt_set = mat_data['gt']
        pan_set = mat_data['pan']
        ms_set = mat_data['ms']
        lms_set = mat_data['lms']
       

        self.gt_set = np.array(gt_set, dtype=np.float32) / 2047
        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047
        self.lms_set = np.array(lms_set, dtype=np.float32) / 2047

    def norm_func(self, x):
        x = x - x.min()
        x = x / x.max()
        x = 2 * x - 1  # [-1, 1]
        return torch.tensor(x, dtype=torch.float32)

    def __getitem__(self, index):
        gt = self.gt_set[index, :, :, :]
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return gt, pan, lms, ms

    def __len__(self):
        return self.gt_set.shape[0]
    # def __len__(self):
    #     return 1