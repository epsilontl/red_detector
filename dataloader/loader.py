import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from dataloader.data_sampler import RandomContinuousSampler


class Loader:
    def __init__(self, dataset, batch_size, num_workers, pin_memory, drop_last, device, shuffle=False, data_index=None):
        self.device = device
        split_indices = list(range(len(dataset)))
        if shuffle:
            # random_list = np.random.choice(range(len(dataset)), len(dataset), replace=False)
            # split_indices = batch_indice(random_list)
            # sampler = RandomContinuousSampler(len(dataset), num=batch_size//batch_size, data_index=data_index)
            sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                                      num_workers=num_workers, pin_memory=pin_memory,
                                                      drop_last=drop_last, collate_fn=collate_events)
        else:
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                      num_workers=num_workers, pin_memory=pin_memory,
                                                      drop_last=drop_last, collate_fn=collate_events_test)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


def collate_events(data):
    batch_labels = []
    batch_histograms = []
    idx_batch = 0

    for d in data:  # different batch
        histograms = []
        for idx in range(len(d[0])):
            label = d[0][idx]
            histogram = d[1][idx]
            lb = np.concatenate([label, idx_batch*np.ones((len(label), 1), dtype=np.float32)], 1)
            batch_labels.append(lb)
            histograms.append(histogram)
            idx_batch += 1

        batch_histograms.append(np.array(histograms))
    labels = torch.from_numpy(np.concatenate(batch_labels, 0))
    histograms = default_collate(batch_histograms)

    return labels, histograms


def collate_events_test(data):
    labels = []
    histograms = []
    idx_batch = 0

    for d in data:
        for idx in range(len(d[0])):
            label = d[0][idx]
            histogram = d[1][idx]

            lb = np.concatenate([label, idx_batch*np.ones((len(label), 1), dtype=np.float32)], 1)
            labels.append(lb)
            histograms.append(histogram)
            idx_batch += 1

    labels = torch.from_numpy(np.concatenate(labels, 0))
    histograms = default_collate(histograms)

    return labels, histograms
