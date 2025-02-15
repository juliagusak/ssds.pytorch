
import torch
import torch.utils.data as data
import numpy as np

from lib.utils.data_augment import preproc

from lib.dataset import voc
from lib.dataset import coco

dataset_map = { 'voc': voc.VOCDetection, 'coco': coco.COCODetection }

def gen_dataset_fn(name):
    """Returns a dataset func.

    Args:
    name: The name of the dataset.

    Returns:
    func: dataset_fn

    Raises:
    ValueError: If network `name` is not recognized.
    """
    if name not in dataset_map:
        raise ValueError('The dataset unknown %s' % name)
    func = dataset_map[name]
    return func



def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            #print('in detection_collate', type(tup), tup.shape)
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


def load_data(cfg, phase):
    if phase == 'train':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TRAIN_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, cfg.PROB))
        data_loader = data.DataLoader(dataset, cfg.TRAIN_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True, drop_last=True)
    if phase == 'eval':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, -1))
        data_loader = data.DataLoader(dataset, cfg.TEST_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    if phase == 'test':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, -2))
        data_loader = data.DataLoader(dataset, cfg.TEST_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    if phase == 'visualize':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, 1))
        data_loader = data.DataLoader(dataset, cfg.TEST_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    return data_loader

def verify_dataset(loader):
    import pdb
    for i, batch in enumerate(loader):
        data, label = batch
        print(data.shape, len(label))
        pdb.set_trace()


if __name__ == '__main__':
    import argparse
    from lib.utils.config_parse import cfg_from_file
    from lib.utils.config_parse import cfg

    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file', help='optional config file', default=None, type=str)
    args = parser.parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
        train_loader = load_data(cfg.DATASET, 'train')
        verify_dataset(train_loader)
    else:
        parser.print_help()



