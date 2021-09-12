import os
import numpy as np
import cv2
from PIL import Image, ImageFile
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import argparse
# from datasets import make_dataloader
# from processor import do_inference
from config import cfg
from model import make_model
# from datasets.vehicleid import VehicleID
# from datasets.bases import ImageDataset
# from utils.metrics import R1_mAP_eval
from utils.reranking import re_ranking

# from utils.logger import setup_logger

ImageFile.LOAD_TRUNCATED_IMAGES = True


def visualize_result(query, out_dir, submit_txt_path, topk=5):
    vis_size = (256, 256)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    results = []
    with open(submit_txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            results.append(line.split(' '))

    for i, result in enumerate(results):
        gallery_paths = []
        for name in result:
            # gallery_paths.append(os.path.join(gallery_dir, index.zfill(6)+'.jpg'))
            gallery_paths.append(os.path.join(gallery_dir, name))

        imgs = []
        imgs.append(cv2.resize(cv2.imread(query[i]), vis_size))
        for n in range(topk):
            img = cv2.resize(cv2.imread(gallery_paths[n]), vis_size)
            imgs.append(img)

        canvas = np.concatenate(imgs, axis=1)
        # if is_False:
        #
        cv2.namedWindow('search_result', 0)
        cv2.imshow('search_result', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #
        cv2.imwrite(os.path.join(out_dir, os.path.basename(query[i])), canvas)


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def collate_fn(batch):
    imgs, img_paths = zip(*batch)  # 解压已打包的元组，返回各元组组成的列表
    return torch.stack(imgs, dim=0), img_paths


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path.split('/')[-1]


class distmat_eval():
    def __init__(self, num_query, feat_norm=True, reranking=False):
        super(distmat_eval, self).__init__()
        self.num_query = num_query
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []

    def update(self, output):  # called once for each batch
        feat = output
        self.feats.append(feat.cpu())

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]

        # gallery
        gf = feats[self.num_query:]

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        # cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return distmat


if __name__ == '__main__':
    cfg_file = "./configs/VehicleID/vit_base.yml"
    query_dir = "./test_data/query"
    gallery_dir = "./test_data/gallery"
    if not os.path.exists(query_dir):
        raise RuntimeError('"{}" is not available'.format(query_dir))
    if not os.path.exists(gallery_dir):
        raise RuntimeError('"{}" is not available'.format(gallery_dir))
    output_dir = "./results/vehicleID_vit_base"
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cfg.merge_from_file(cfg_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # num_workers = cfg.DATALOADER.NUM_WORKERS
    # dataset = VehicleID(root="./data")
    # num_classes = dataset.num_train_pids  # kinds of vehicles and for the VehicleID dataset it should be 13164
    # cam_num = dataset.num_train_cams  # here should be 1
    # view_num = dataset.num_train_vids  # here should be 1
    # query_num = len(dataset.query)
    model = make_model(cfg, num_class=13164, camera_num=1, view_num=1)  # for VehicleID, the num_class is 13164
    model.load_param('./vit_base_vehicleID.pth')
    device = "cuda"
    if device:
        model.to(device)

    query_data = []
    for file in os.listdir(query_dir):
        if file.endswith(('jpg', 'png', 'jpeg', 'bmp')):
            query_data.append(os.path.join(query_dir, file))
    query_num = len(query_data)
    gallery_data = []
    for file in os.listdir(gallery_dir):
        if file.endswith(('jpg', 'png', 'jpeg', 'bmp')):
            gallery_data.append(os.path.join(gallery_dir, file))
    # val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    transforms = T.Compose([
        # T.Resize([256, 256]),
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageDataset(query_data + gallery_data, transforms)
    num_workers = 8
    dataloader = DataLoader(
        dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )
    # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    # model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num=view_num)
    # model.load_param(cfg.TEST.WEIGHT)

    # do_inference(cfg, model, val_loader, len(dataset.query))

    evaluator = distmat_eval(query_num, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    model.eval()
    img_path_list = []

    for n_iter, (img, imgpath) in enumerate(dataloader):
        with torch.no_grad():
            img = img.to(device)
            feat = model(img)
            evaluator.update(feat)
            img_path_list.extend(imgpath)

    dist_mat = evaluator.compute()
    #
    g_imgs = img_path_list[query_num:]
    num_q, _ = dist_mat.shape
    indices = np.argsort(dist_mat, axis=1)
    with open('result.txt', 'w') as output:
        for q_idx in range(num_q):
            order = indices[q_idx]
            row = [g_imgs[i] for i in order]
            output.write(' '.join([os.path.basename(val) for val in row]) + '\n')
    visualize_result(query_data, "./results", "./result.txt")
    #
