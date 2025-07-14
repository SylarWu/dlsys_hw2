import gzip
import struct
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'rb') as image_file:
            data_magic, data_nums, data_rows, data_cols = struct.unpack('>IIII', image_file.read(16))
            print(f"image data meta info: Magic:{data_magic}, Nums:{data_nums}, Rows:{data_rows}, Cols:{data_cols}")
            data_buffer = image_file.read(data_nums * data_rows * data_cols)
            data = np.frombuffer(data_buffer, dtype=np.uint8)
            data = np.reshape(data, (data_nums, data_rows, data_cols, 1)).astype(np.float32)
            data = data / 255.0
        with (gzip.open(label_filename, 'rb')) as label_file:
            label_magic, label_nums = struct.unpack('>II', label_file.read(8))
            print(f"image label meta info: Magic:{label_magic}, Nums:{label_nums}")
            label_buffer = label_file.read(label_nums)
            label = np.frombuffer(label_buffer, dtype=np.uint8)
        self.data = data
        self.label = label
        self.transforms = transforms if transforms is not None else []
        self.num_samples = len(self.data)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        sample_data, sample_label = self.data[index], self.label[index]
        for transformation in self.transforms:
            sample_data = transformation(sample_data)
        return sample_data, sample_label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.num_samples
        ### END YOUR SOLUTION