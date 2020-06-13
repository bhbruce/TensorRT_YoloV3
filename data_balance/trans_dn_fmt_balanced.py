import os
import shutil
import random
import collections
import pandas as pd
from scipy.io import loadmat
from PIL import Image
from argparse import ArgumentParser


def get_gt(mat_data):
    # id: 0, objs: 3
    items = [(annots[0][0][0][0], annots[0][0][0][3]) for annots in mat_data]
    gt = []
    for item in items:
        objs = []
        for obj in item[1]:
            objs.append([obj[0][0][0][0][0], obj[0][0][0][1][0]])
        gt.append((item[0][0], objs))

    return gt


def get_img_size(img_path, gt):
    img_full_names = [
            os.path.realpath(os.path.join(img_path, item[0])) for item in gt]

    return [Image.open(img).size for img in img_full_names]


def bbox_norm(s, gt, l):
    for idx in range(len(s)):
        objs = gt[idx][1]
        for obj in objs:
            # xmin     , ymin     , xmax     , ymax
            # obj[1][0], obj[1][1], obj[1][2], obj[1][3]
            # bbox = obj[1], idx_size = s[idx]

            # width = (x_max - x_min) / idx_size[0]
            width = float((obj[1][2] - obj[1][0]))
            # height = (y_max - y_min) / idx_size[1]
            height = float((obj[1][3] - obj[1][1]))
            # x_center = (x_min + width/2) / idx_size[0]
            x_center = float((obj[1][0] + width/2)) / s[idx][0]
            # y_center = (y_min + height/2) / idx_size[1]
            y_center = float((obj[1][1] + height/2)) / s[idx][1]

            # Update normalized bbox and binary label
            obj[1] = (x_center, y_center, width/s[idx][0], height/s[idx][1])
            obj[0] = l.index(obj[0])

    return gt


def find_labels(gt):
    checker = set()
    for img in gt:
        for item in img[1]:
            checker.add(item[0])

    return sorted(checker)


def check_dir(p):
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)


def filt_cls(gts):
    filt_list = ['phone', 'coat', 'surfboard', 'watch', 'trees',
                 'ball', 'bag', 'sofa', 'glasses', 'roof', 'boat']
    rm_lst = []
    for gt in gts:
        for bbox in gt[1]:
            if bbox[0] in filt_list:
                rm_lst.append(gt)
                break
    for rm_elem in rm_lst:
        gts.remove(rm_elem)

    return gts


def cat_dist(gts):
    """Calculate counts distribution of each categories"""
    counter = collections.Counter()
    for img, gt in gts:
        for name, bbox in gt:
            counter.update([name])

    return pd.Series(counter).sort_values(ascending=False)


def split_ds(gts, r=0.2):
    def tmp_cal(idxes):
        c = collections.defaultdict(lambda: 0)
        for i in idxes:
            for n, _ in gts[i][1]:
                c[n] += 1
        return c

    lbl_lst = cat_dist(gts)
    # Exclude top 3 classes, reverse
    balance_cls = lbl_lst[::-1]
    upper_bound = int(len(gts) * r)
    val_idx = set()
    added_lst = []
    for cls, counter in balance_cls.items():
        counter = int(counter * r) - tmp_cal(val_idx)[cls]
        for idx, (_, gt) in enumerate(gts):
            names = [n for n, _ in gt]
            include_added = bool(set(names).intersection(added_lst))
            if (counter <= 0) or (len(val_idx) > upper_bound):
                break
            elif (not include_added) and cls in names:
                val_idx.add(idx)
                counter -= sum([True for n in names if n == cls])
        added_lst.append(cls)

    val_ds = []
    for idx in val_idx:
        val_ds.append(gts[idx])

    return val_ds


def dump_labels(gt, lbl_path):
    check_dir(lbl_path)
    # item[0]: img_name, item[1]: annots
    for item in gt:
        path = os.path.join(lbl_path, item[0].split(".")[0]+".txt")
        with open(path, "w") as f:
            for obj in item[1]:
                _ = f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    obj[0], obj[1][0], obj[1][1], obj[1][2], obj[1][3]))


def dump_names(l, path):
    with open(path, "w") as f:
        for name in l:
            _ = f.write(f"{name}\n")


def dump_ds_file(gts, t_file_path, v_file_path, img_path):
    val_ds = split_ds(gts)
    train_ds = [gt for gt in gts if gt not in val_ds]
    random.seed(82)
    random.shuffle(train_ds)
    print(len(gts))
    print(len(train_ds))
    print(len(val_ds))
    with open(t_file_path, "w") as f:
        for img in train_ds:
            _ = f.write("{}\n".format(os.path.join(img_path, img[0])))

    with open(v_file_path, "w") as f:
        for img in val_ds:
            _ = f.write("{}\n".format(os.path.join(img_path, img[0])))


def dump_data_file(path, cls_len, train, val, name):
    with open(path, "w") as f:
        _ = f.write(f"classes={cls_len}\n")
        _ = f.write(f"train={os.path.realpath(train)}\n")
        _ = f.write(f"valid={os.path.realpath(val)}\n")
        _ = f.write(f"names={os.path.realpath(name)}\n")


def run(params):
    if not os.path.exists("./data"):
        raise ValueError("\"./data\" does not exist.")
    if params['data_dir'] is None:
        raise ValueError("Please provide dataset directory")
    mat_path = os.path.join(params['data_dir'], "annotations.mat")
    img_path = os.path.join(params['data_dir'], "images/")
    labels_path = os.path.join(params['data_dir'], "labels/")
    names_file_path = "./data/unrel.names"
    train_file_path = "./data/unrel_train.txt"
    val_file_path = "./data/unrel_val.txt"
    data_file_path = "./data/unrel.data"

    mat_data = loadmat(mat_path)
    ground_truth = get_gt(mat_data['annotations'])
    ground_truth = filt_cls(ground_truth)

    img_size = get_img_size(img_path, ground_truth)
    label_list = find_labels(ground_truth)
    normalized_gt = bbox_norm(img_size, ground_truth, label_list)

    if params['label_dir']:
        dump_labels(normalized_gt, labels_path)
    dump_names(label_list, names_file_path)
    dump_ds_file(normalized_gt, train_file_path, val_file_path, img_path)
    dump_data_file(data_file_path, len(label_list),
                   train_file_path, val_file_path, names_file_path)


def param_loader():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        help="Path to dataset.")
    parser.add_argument("--label_dir", type=str, default=None,
                        help="Path to label dir.")
    args, _ = parser.parse_known_args()
    return vars(args)


if __name__ == "__main__":
    p = param_loader()
    run(p)
