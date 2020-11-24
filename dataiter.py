from torch.utils.data import Dataset
import numpy as np

class TensorDataset3(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing the three tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets.
        label_tensor (Tensor): contains sample labels
    """

    def __init__(self, data_tensor, target_tensor, lsh, label_tensor):
        print('Data1:', data_tensor.size())
        print('Data2:', target_tensor.size())
        print('Labels Hash Code:', lsh.size())
        print('Labels:', label_tensor.size())
        assert data_tensor.size(0) == target_tensor.size(0)
        assert data_tensor.size(0) == label_tensor.size(0)
        assert data_tensor.size(0) == lsh.size(0)

        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.training_full = lsh
        self.label_tensor = label_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.training_full[index], self.label_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)