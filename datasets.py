import os, re, random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm.notebook import tqdm

class PairDataset(Dataset):
    """
    Deterministic pair dataset:
    - For each person ID, generates `num_pairs_per_id` pairs.
    - Each set of pairs: 50% positive, 50% negative.
    - Length = num_pairs_per_id * num_persons.
    """

    def __init__(self, person_to_images, transform=None, num_pairs_per_id=10):
        self.person_to_images = person_to_images
        self.person_ids = list(person_to_images.keys())
        self.transform = transform
        self.num_pairs_per_id = num_pairs_per_id
        self.total_len = len(self.person_ids) * num_pairs_per_id

    def __len__(self):
        return self.total_len

    def _read_img(self, path):
        img = read_image(str(path)).float() / 255.0
        return img

    def __getitem__(self, idx):
        pid_idx = idx // self.num_pairs_per_id
        pair_idx = idx % self.num_pairs_per_id
        pid = self.person_ids[pid_idx]

        # The first half pairs are positive (the same person), the other half is negative
        positive = pair_idx < (self.num_pairs_per_id // 2)

        if positive:
            imgs = self.person_to_images[pid]
            img1_path, img2_path = random.sample(imgs, 2)
            label = 1
        else:
            img1_path = random.choice(self.person_to_images[pid])
            pid2 = random.choice([p for p in self.person_ids if p != pid])
            img2_path = random.choice(self.person_to_images[pid2])
            label = 0

        img1 = self._read_img(img1_path)
        img2 = self._read_img(img2_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


class PairFeatureDataset(torch.utils.data.Dataset):
    """
    Pre-computation of the features per image, using a provided extractor.
    Much faster as features are computed once, with batched extractor calls.
    """
    def __init__(self, base_dataset, extractor, device, batch_size=8):
        feats1, feats2, labels = [], [], []
        extractor.eval().to(device)

        loader = torch.utils.data.DataLoader(
            base_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        with torch.no_grad():
            for imgs1, imgs2, lbls in tqdm(loader, desc="Processing"):
                imgs1 = imgs1.to(device, non_blocking=True)
                imgs2 = imgs2.to(device, non_blocking=True)

                f1 = extractor(imgs1).cpu()
                f2 = extractor(imgs2).cpu()

                feats1.append(f1)
                feats2.append(f2)
                labels.append(lbls)

        self.f1s = torch.cat(feats1, dim=0)
        self.f2s = torch.cat(feats2, dim=0)
        self.labels = torch.cat(labels, dim=0)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return self.f1s[idx], self.f2s[idx], self.labels[idx]



def make_feature_datasets(images_dataset, extractor, device):
    """
    Convert image-based PairDatasets into feature-based PairFeatureDatasets
    """
    features_dataset = PairFeatureDataset(images_dataset, extractor, device)
    return features_dataset



class CUHK03PairDataset(PairDataset):
    def __init__(self, root_dir, transform=None, num_pairs_per_id=10):
        person_to_images = {}
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(".jpg"):
                pid = filename.split("_")[0]
                person_to_images.setdefault(pid, []).append(os.path.join(root_dir, filename))
        super().__init__(person_to_images, transform, num_pairs_per_id)


class MarketPairDataset(PairDataset):
    def __init__(self, root_dir, transform=None, num_pairs_per_id=10):
        person_to_images = {}
        for filename in os.listdir(root_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                pid = filename.split("_")[0]
                if pid == "-1":
                    continue
                person_to_images.setdefault(pid, []).append(os.path.join(root_dir, filename))
        super().__init__(person_to_images, transform, num_pairs_per_id)
