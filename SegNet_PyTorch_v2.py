# SegNet PyTorch - Final training script for ISPRS Vaihingen
# Generates:
# results_segnet/
#   results_segnet.txt
#   metrics.csv
#   checkpoints/
#   predictions/

import os
import csv
import random
import itertools
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =========================
# Configuration
# =========================

DATASET = "Vaihingen"

# Change this if your dataset is located elsewhere
ROOT = r"C:\Users\denisa.barb\Desktop\Disertatie\ISPRS_semantic_labeling_Vaihingen"

DATA_FOLDER = os.path.join(ROOT, "top", "top_mosaic_09cm_area{}.tif")
LABEL_FOLDER = os.path.join(ROOT, "gts_for_participants", "top_mosaic_09cm_area{}.tif")
ERODED_FOLDER = LABEL_FOLDER

WINDOW_SIZE = (256, 256)
BATCH_SIZE = 4
CACHE = False
NUM_CLASSES = 6
IN_CHANNELS = 3
EPOCHS = 100
SAVE_EPOCH = 10
BASE_LR = 0.01

TRAIN_IDS =  [
    '1', '2', '3', '4', '6', '7', '8', '9', '10',
    '11', '12', '13', '14', '16', '17', '18', '19',
    '20', '22', '23', '24', '25', '26', '27', '28',
    '29', '31', '32', '33', '34', '35', '36', '37'
]
TEST_IDS = ['5', '21', '15', '30']

RESULTS_DIR = "results_segnet"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
RESULTS_TXT = os.path.join(RESULTS_DIR, "results_segnet.txt")
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics.csv")

for folder in [RESULTS_DIR, CHECKPOINT_DIR, PREDICTIONS_DIR]:
    os.makedirs(folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# ISPRS Palette
# =========================

palette = {
    0: (255, 255, 255),  # Impervious surfaces
    1: (0, 0, 255),      # Buildings
    2: (0, 255, 255),    # Low vegetation
    3: (0, 255, 0),      # Trees
    4: (255, 255, 0),    # Cars
    5: (255, 0, 0),      # Clutter/background
    6: (0, 0, 0)         # Undefined
}

invert_palette = {v: k for k, v in palette.items()}

CLASS_NAMES = [
    "Impervious surfaces",
    "Buildings",
    "Low vegetation",
    "Trees",
    "Cars",
    "Clutter/background"
]

WEIGHTS = torch.ones(NUM_CLASSES)


def convert_to_color(arr_2d, palette=palette):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, color in palette.items():
        if c >= NUM_CLASSES:
            continue
        arr_3d[arr_2d == c] = color
    return arr_3d


def convert_from_color(arr_3d, palette=invert_palette):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for color, cls in palette.items():
        if cls >= NUM_CLASSES:
            continue
        mask = np.all(arr_3d == np.array(color).reshape(1, 1, 3), axis=2)
        arr_2d[mask] = cls
    return arr_2d


# =========================
# Utility functions
# =========================

def get_random_pos(img, window_shape):
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def cross_entropy_2d(input_tensor, target, weight=None):
    if input_tensor.dim() == 2:
        return F.cross_entropy(input_tensor, target, weight=weight)
    if input_tensor.dim() == 4:
        output = input_tensor.view(input_tensor.size(0), input_tensor.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight=weight)
    raise ValueError("Expected 2 or 4 dimensions, got {}".format(input_tensor.dim()))


def pixel_accuracy(pred, target):
    return 100.0 * float(np.count_nonzero(pred == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    count = 0
    for _ in sliding_window(top, step, window_size):
        count += 1
    return count


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def compute_class_iou(pred, target, num_classes=NUM_CLASSES):
    pred = np.asarray(pred)
    target = np.asarray(target)
    ious = []

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)

    return np.array(ious, dtype=np.float32)


def save_comparison_png(rgb, gt, pred, save_path):
    rgb = np.asarray(rgb)

    if rgb.max() <= 1.0:
        rgb = (255 * rgb).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)

    gt_color = convert_to_color(gt)
    pred_color = convert_to_color(pred)

    h, w = gt_color.shape[:2]
    if rgb.shape[:2] != (h, w):
        import cv2
        rgb = cv2.resize(rgb, (w, h))

    comparison = np.concatenate([rgb, gt_color, pred_color], axis=1)
    io.imsave(save_path, comparison)


def init_results_files():
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write("SegNet training results\n")
        f.write("Created: {}\n".format(datetime.now()))
        f.write("Dataset: {}\n".format(DATASET))
        f.write("Training tiles: {}\n".format(TRAIN_IDS))
        f.write("Validation tiles: {}\n".format(TEST_IDS))
        f.write("Batch size: {}\n".format(BATCH_SIZE))
        f.write("Window size: {}\n".format(WINDOW_SIZE))
        f.write("Device: {}\n".format(device))
        if torch.cuda.is_available():
            f.write("GPU: {}\n".format(torch.cuda.get_device_name(0)))
        f.write("----------------------------------------\n")

    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_pixel_accuracy",
            "val_pixel_accuracy",
            "miou",
            "iou_impervious",
            "iou_buildings",
            "iou_low_vegetation",
            "iou_trees",
            "iou_cars",
            "iou_clutter"
        ])


def save_epoch_results(epoch, train_loss, train_acc, val_acc, class_iou):
    miou = np.nanmean(class_iou)

    with open(RESULTS_TXT, "a", encoding="utf-8") as f:
        f.write("========== Epoch {} ==========\n".format(epoch))
        f.write("Training loss: {:.6f}\n".format(train_loss))
        f.write("Training pixel accuracy: {:.6f}\n".format(train_acc))
        f.write("Validation pixel accuracy: {:.6f}\n".format(val_acc))
        f.write("mIoU: {:.6f}\n".format(miou))
        f.write("Class-wise IoU:\n")
        for name, value in zip(CLASS_NAMES, class_iou):
            f.write("  {:22s}: {:.6f}\n".format(name, value if not np.isnan(value) else -1))
        f.write("----------------------------------------\n\n")

    with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            train_loss,
            train_acc,
            val_acc,
            miou,
            *class_iou
        ])


# =========================
# Dataset class
# =========================

class ISPRSDataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super().__init__()
        self.augmentation = augmentation
        self.cache = cache

        self.data_files = [data_files.format(tile_id) for tile_id in ids]
        self.label_files = [label_files.format(tile_id) for tile_id in ids]

        for path in self.data_files + self.label_files:
            if not os.path.isfile(path):
                raise FileNotFoundError("{} is not a file.".format(path))

        self.data_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        return 10000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip = flip and random.random() < 0.5
        will_mirror = mirror and random.random() < 0.5

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, index):
        random_idx = random.randint(0, len(self.data_files) - 1)

        if random_idx in self.data_cache_:
            data = self.data_cache_[random_idx]
        else:
            data = 1 / 255 * np.asarray(
                io.imread(self.data_files[random_idx]).transpose((2, 0, 1)),
                dtype="float32"
            )
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.label_cache_:
            label = self.label_cache_[random_idx]
        else:
            label = np.asarray(
                convert_from_color(io.imread(self.label_files[random_idx])),
                dtype="int64"
            )
            if self.cache:
                self.label_cache_[random_idx] = label

        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        if self.augmentation:
            data_p, label_p = self.data_augmentation(data_p, label_p)

        return torch.from_numpy(data_p), torch.from_numpy(label_p)


# =========================
# SegNet architecture
# =========================

class SegNet(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)

    def __init__(self, in_channels=IN_CHANNELS, out_channels=NUM_CLASSES):
        super().__init__()

        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)

        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)

        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)

        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)

        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)

        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)

        self.apply(self.weight_init)

    def forward(self, x):
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)

        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)

        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)

        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)

        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)

        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = self.conv1_1_D(x)
        return x


# =========================
# VGG16 encoder initialization
# =========================

def load_vgg16_weights(net):
    vgg_path = "./vgg16_bn-6c64b313.pth"
    if not os.path.isfile(vgg_path):
        print("VGG16 weights not found. Encoder will use random initialization.")
        return

    print("Loading VGG16 weights:", vgg_path)
    vgg16_weights = torch.load(vgg_path, map_location=device)
    own_state = net.state_dict()

    mapped = {}
    for k_vgg, k_segnet in zip(vgg16_weights.keys(), own_state.keys()):
        if "features" in k_vgg and own_state[k_segnet].shape == vgg16_weights[k_vgg].shape:
            mapped[k_segnet] = vgg16_weights[k_vgg]

    own_state.update(mapped)
    net.load_state_dict(own_state)
    print("Loaded {} VGG16-compatible tensors into SegNet.".format(len(mapped)))


# =========================
# Evaluation
# =========================

def test(net, test_ids, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE,
         window_size=WINDOW_SIZE, save_epoch=None, save_predictions=True):
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(tile_id)), dtype="float32")
                   for tile_id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(tile_id)), dtype="uint8")
                   for tile_id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(tile_id)))
                     for tile_id in test_ids)

    all_preds = []
    all_gts = []

    net.eval()

    with torch.no_grad():
        for tile_idx, (img, gt, gt_e) in enumerate(
            tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False)
        ):
            pred = np.zeros(img.shape[:2] + (NUM_CLASSES,))

            total = max(1, count_sliding_window(img, step=stride, window_size=window_size) // batch_size)

            for coords in tqdm(
                grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)),
                total=total,
                leave=False
            ):
                image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = torch.from_numpy(image_patches).to(device)

                outs = net(image_patches)
                outs = outs.detach().cpu().numpy()

                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x+w, y:y+h] += out

            pred = np.argmax(pred, axis=-1)

            all_preds.append(pred)
            all_gts.append(gt_e)

            if save_predictions and save_epoch is not None and tile_idx == 0:
                save_path = os.path.join(PREDICTIONS_DIR, "epoch_{}_prediction.png".format(save_epoch))
                save_comparison_png(np.asarray(255 * img, dtype="uint8"), gt_e, pred, save_path)

    flat_preds = np.concatenate([p.ravel() for p in all_preds])
    flat_gts = np.concatenate([g.ravel() for g in all_gts])

    val_acc = pixel_accuracy(flat_preds, flat_gts)
    class_iou = compute_class_iou(flat_preds, flat_gts, NUM_CLASSES)

    print("Validation pixel accuracy: {:.6f}".format(val_acc))
    print("Validation mIoU: {:.6f}".format(np.nanmean(class_iou)))

    for name, value in zip(CLASS_NAMES, class_iou):
        print("  {}: {:.6f}".format(name, value if not np.isnan(value) else -1))

    return val_acc, class_iou


# =========================
# Training
# =========================

def train(net, optimizer, train_loader, epochs=EPOCHS, scheduler=None,
          weights=WEIGHTS, save_epoch=SAVE_EPOCH):
    init_results_files()

    weights = weights.to(device)
    iter_ = 0

    for epoch in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()

        net.train()

        epoch_losses = []
        epoch_accs = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = net(data)

            loss = cross_entropy_2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            pred = np.argmax(output.detach().cpu().numpy()[0], axis=0)
            gt = target.detach().cpu().numpy()[0]

            batch_acc = pixel_accuracy(pred, gt)
            epoch_losses.append(loss.item())
            epoch_accs.append(batch_acc)

            if iter_ % 100 == 0:
                print(
                    "Train (epoch {}/{}) [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.4f}".format(
                        epoch,
                        epochs,
                        batch_idx,
                        len(train_loader),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        batch_acc
                    )
                )

            iter_ += 1

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(np.mean(epoch_accs))

        if epoch == 1 or epoch % save_epoch == 0:
            val_acc, class_iou = test(
                net,
                TEST_IDS,
                stride=min(WINDOW_SIZE),
                save_epoch=epoch,
                save_predictions=True
            )

            save_epoch_results(epoch, train_loss, train_acc, val_acc, class_iou)

            checkpoint_path = os.path.join(CHECKPOINT_DIR, "segnet_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), checkpoint_path)
            print("Saved checkpoint:", checkpoint_path)

    final_path = os.path.join(CHECKPOINT_DIR, "segnet_final.pth")
    torch.save(net.state_dict(), final_path)
    print("Training finished. Final model saved to:", final_path)


# =========================
# Main
# =========================

def main():
    print("Training tiles:", TRAIN_IDS)
    print("Validation tiles:", TEST_IDS)

    train_set = ISPRSDataset(TRAIN_IDS, cache=CACHE)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    net = SegNet().to(device)
    load_vgg16_weights(net)

    params = []
    for key, value in dict(net.named_parameters()).items():
        if "_D" in key:
            params += [{"params": [value], "lr": BASE_LR}]
        else:
            params += [{"params": [value], "lr": BASE_LR / 2}]

    optimizer = optim.SGD(params, lr=BASE_LR, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)

    train(net, optimizer, train_loader, EPOCHS, scheduler=scheduler, save_epoch=SAVE_EPOCH)


if __name__ == "__main__":
    main()
