import torch
from dataprep import *
from torch.nn import functional as F
from torch.utils.data import Dataset

class LibriSpeechDataset(Dataset):
    def __init__(self, feat_scp, feat_mean=None, feat_var=None, shuffling=False, rand=None):
        f = open(feat_scp)
        self.feat_entries = [parse_scp_line(line) for line in f.readlines()]
        f.close()

        self.feat_mean = feat_mean
        self.feat_var = feat_var

        self.indices = list(range(len(self.feat_entries)))

        if shuffling:
            rand.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.feat_entries)

    def __getitem__(self, idx):
        key, file, shift = self.feat_entries[idx]
        feat = load_feat(file, shift, self.feat_mean, self.feat_var)
        feat = torch.Tensor(feat)

        length = feat.shape[0]

        return feat, length

class LibriSpeech:
    def __init__(self, feat_scp, feat_mean=None, feat_var=None, shuffling=False, rand=None):
        f = open(feat_scp)
        self.feat_entries = [parse_scp_line(line) for line in f.readlines()]
        f.close()

        self.feat_mean = feat_mean
        self.feat_var = feat_var

        self.indices = list(range(len(self.feat_entries)))

        if shuffling:
            rand.shuffle(self.indices)

    def __iter__(self):
        for i in self.indices:
            key, file, shift = self.feat_entries[i]
            feat = load_feat(file, shift, self.feat_mean, self.feat_var)

            yield key, feat

class WsjFeat:
    def __init__(self, feat_scp, feat_mean=None, feat_var=None, shuffling=False, rand=None):
        f = open(feat_scp)
        self.feat_entries = [parse_scp_line(line) for line in f.readlines()]
        f.close()

        self.feat_mean = feat_mean
        self.feat_var = feat_var

        self.indices = list(range(len(self.feat_entries)))

        if shuffling:
            rand.shuffle(self.indices)

    def __iter__(self):
        for i in self.indices:
            feat_key, feat_file, feat_shift = self.feat_entries[i]
            feat = load_feat(feat_file, feat_shift, self.feat_mean, self.feat_var)

            yield feat_key, feat

class WsjDataset(Dataset):
    def __init__(self, feat_scp, label_scp, label_dict, feat_mean=None, feat_var=None, shuffling=False, rand=None):
        f = open(feat_scp)
        self.feat_entries = [parse_scp_line(line) for line in f.readlines()]
        f.close()

        f = open(label_scp)
        self.label_entries = [parse_scp_line(line) for line in f.readlines()]
        f.close()

        assert len(feat_scp) == len(label_scp)

        self.label_dict = label_dict
        self.feat_mean = feat_mean
        self.feat_var = feat_var

        self.indices = list(range(len(self.feat_entries)))

        if shuffling:
            rand.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.feat_entries)

    def __getitem__(self, idx):
        feat_key, feat_file, feat_shift = self.feat_entries[idx]
        feat = load_feat(feat_file, feat_shift, self.feat_mean, self.feat_var)
        feat = torch.Tensor(feat)

        feat_length = feat.shape[0]

        label_key, label_file, label_shift = self.label_entries[idx]
        labels = load_labels(label_file, label_shift)
        labels = torch.Tensor([self.label_dict[c] for c in labels])

        label_length = labels.shape[0]

        assert feat_key == label_key

        return feat, labels, feat_length, label_length

class Wsj:
    def __init__(self, feat_scp, label_scp, feat_mean=None, feat_var=None, shuffling=False, rand=None):
        f = open(feat_scp)
        self.feat_entries = [parse_scp_line(line) for line in f.readlines()]
        f.close()

        f = open(label_scp)
        self.label_entries = [parse_scp_line(line) for line in f.readlines()]
        f.close()

        assert len(feat_scp) == len(label_scp)

        self.feat_mean = feat_mean
        self.feat_var = feat_var

        self.indices = list(range(len(self.feat_entries)))

        if shuffling:
            rand.shuffle(self.indices)

    def __iter__(self):
        for i in self.indices:
            feat_key, feat_file, feat_shift = self.feat_entries[i]
            feat = load_feat(feat_file, feat_shift, self.feat_mean, self.feat_var)

            label_key, label_file, label_shift = self.label_entries[i]
            labels = load_labels(label_file, label_shift)

            assert feat_key == label_key

            yield feat_key, feat, labels


