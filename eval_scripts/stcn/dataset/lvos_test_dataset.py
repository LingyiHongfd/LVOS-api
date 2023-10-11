import os
from os import path
import json

from torch.utils.data.dataset import Dataset

from dataset.video_reader import VideoReader


class LVOSTestDataset:
    def __init__(self, data_root, split, size=480):
        self.image_dir = path.join(data_root, 'JPEGImages')
        if split=='val':
            self.mask_dir = path.join(data_root, 'Annotations_convert')
        else:
            self.mask_dir = path.join(data_root, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video,
                              path.join(self.image_dir, video),
                              path.join(self.mask_dir, video),
                              size=self.size,
                              use_all_mask=True
                              )

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        video = self.vid_list[idx]
        return VideoReader(video,
                           path.join(self.image_dir, video),
                           path.join(self.mask_dir, video),
                           size=self.size,
                           use_all_mask=True
                           )
