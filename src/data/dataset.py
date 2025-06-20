import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TrainingDataset(Dataset):
    def __init__(self, imgs, pairs, pair_labels, mean, std):
        self.imgs = imgs
        self.pairs = pairs
        self.pair_labels = torch.tensor(pair_labels, dtype=torch.int64)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        idx1, idx2 = self.pairs[index]

        img1 = self.imgs[idx1]
        img2 = self.imgs[idx2]

        img1 = self.transform(img1.transpose(1,2,0))
        img2 = self.transform(img2.transpose(1,2,0))

        label = self.pair_labels[index]

        return img1, img2, label, idx1, idx2

