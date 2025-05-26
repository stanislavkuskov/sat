import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import timm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from typing import List
from segmentation_models_pytorch.encoders._base import EncoderMixin
from datetime import datetime
import torch.nn.functional as F
import matplotlib.patches as patches

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
import random
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CRF imports
import cv2

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PVTV2Encoder(torch.nn.Module, EncoderMixin):

    def __init__(self, **kwargs):
        super().__init__()

        # Load PVTv2-B2 model
        self.model = timm.create_model('pvt_v2_b5', pretrained=True, features_only=True)
        
        # Get the output channels from the model
        channels = self.model.feature_info.channels()
        
        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels = channels  # [64, 128, 320, 512]

        # A number of stages in decoder (in other words number of downsampling operations)
        self._depth = 4  # 4 уровня из PVTv2

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels = 3

        # Define output stride
        self._output_stride = 32

    @property
    def output_stride(self):
        return self._output_stride

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Produce list of features of different spatial resolutions."""
        # Get features from the model (x is already normalized by albumentations)
        features = self.model(x)
        return features

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.model.load_state_dict(state_dict, *args, **kwargs)

# Регистрируем энкодер в segmentation_models_pytorch
smp.encoders.encoders["pvt_v2_b5"] = {
    "encoder": PVTV2Encoder,  # encoder class here
    "pretrained_settings": {
        "imagenet": {
            "input_space": "RGB",
            "input_range": [0, 1],
            "repo_id": "timm"  # Add repo_id for timm models
        }
    },
    "params": {
        "depth": 4  # 4 уровня из PVTv2
    }
}

model = smp.FPN(
    encoder_name="pvt_v2_b5",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

# Tiny baseline model
# model = smp.UnetPlusPlus(
#     encoder_name="efficientnet-b2", # mobilenet_v2
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=1
# )

# Large baseline model
# aux_params=dict(
#     pooling='avg',             # one of 'avg', 'max'
#     dropout=0.5,               # dropout ratio, default is None
#     # activation='sigmoid',      # activation function, default is None
#     classes=1,                 # define number of output labels
#     # in_channels=3,
# )

# model = smp.Segformer(
#     encoder_name="mit_b5", # mobilenet_v2
#     encoder_weights="imagenet",
#     # aux_params=aux_params,
#     in_channels=3,
#     classes=1,
# )

# Определяем устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


class SatelliteDataset(Dataset):
    def __init__(self, 
                 root_dirs, 
                 patch_size=512, 
                 stride=512, 
                 transform=None, 
                 split='train', 
                 test_ratio=0.1, 
                 random_seed=42):
        """
        root_dirs: List of paths to scene folders (e.g., ['data/ventura', 'data/santa_rosa'])
        patch_size: Patch size (default 512)
        stride: Stride for patch extraction (default 512)
        transform: Augmentations/transforms
        split: 'train', 'val', or 'test'
        test_ratio: Ratio of test set (default 0.1)
        random_seed: For reproducibility
        """
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.split = split
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        # Read all images and masks
        self.imgs = []
        self.masks = []
        for scene_dir in self.root_dirs:
            img = self._read_rgb(scene_dir)
            mask = self._read_mask(scene_dir)
            assert img.shape[:2] == mask.shape[:2], f"Image and mask size mismatch in {scene_dir}!"
            self.imgs.append(img)
            self.masks.append(mask)

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Create patches list for all scenes
        self.patches = []  # (scene_idx, y, x)
        for scene_idx, img in enumerate(self.imgs):
            h, w = img.shape[:2]
            
            # Делаем размер ячейки в 3-4 раза больше patch_size, кратным stride
            min_cell_size = 3 * patch_size
            cell_size = ((min_cell_size + stride - 1) // stride) * stride
            
            # Вычисляем количество ячеек сетки
            grid_h = max(2, (h - patch_size) // cell_size + 1)  # минимум 2x2 сетка
            grid_w = max(2, (w - patch_size) // cell_size + 1)
            
            # Корректируем размер ячейки, чтобы равномерно покрыть изображение
            cell_h = (h - patch_size) // grid_h + 1
            cell_w = (w - patch_size) // grid_w + 1
            # Округляем до кратного stride
            cell_size_h = ((cell_h + stride - 1) // stride) * stride
            cell_size_w = ((cell_w + stride - 1) // stride) * stride
            
            # Randomly select grid cells for test set
            n_cells = grid_h * grid_w
            n_test_cells = max(2, int(n_cells * self.test_ratio))  # минимум 2 ячейки для теста
            all_cells = [(i, j) for i in range(grid_h) for j in range(grid_w)]
            np.random.shuffle(all_cells)
            test_cells = set(all_cells[:n_test_cells])
            
            # Generate patches based on split
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    # Проверяем все четыре угла патча
                    patch_corners = [
                        (y // cell_size_h, x // cell_size_w),  # верхний левый
                        (y // cell_size_h, (x + patch_size - 1) // cell_size_w),  # верхний правый
                        ((y + patch_size - 1) // cell_size_h, x // cell_size_w),  # нижний левый
                        ((y + patch_size - 1) // cell_size_h, (x + patch_size - 1) // cell_size_w)  # нижний правый
                    ]
                    
                    # Проверяем, все ли углы патча находятся в одном типе ячеек (тест или не тест)
                    corners_in_test = sum(1 for corner in patch_corners if corner in test_cells)
                    
                    if split == 'test':
                        # Для тестового набора все углы должны быть в тестовых ячейках
                        if corners_in_test == len(patch_corners):
                            self.patches.append((scene_idx, y, x))
                    else:  # train/val
                        # Для train/val ни один угол не должен быть в тестовых ячейках
                        if corners_in_test == 0:
                            self.patches.append((scene_idx, y, x))

        # For train/val split, randomly assign patches
        if split != 'test':
            np.random.shuffle(self.patches)
            n_total = len(self.patches)
            n_train = int(n_total * 0.8)  # 80% for training
            if split == 'train':
                self.patches = self.patches[:n_train]
            else:  # val
                self.patches = self.patches[n_train:]  # remaining 20% for validation

    def _read_rgb(self, scene_dir):
        r = self._read_tif(os.path.join(scene_dir, 'RED.tif'))
        g = self._read_tif(os.path.join(scene_dir, 'GRN.tif'))
        b = self._read_tif(os.path.join(scene_dir, 'BLUE.tif'))
        rgb = np.stack([r, g, b], axis=-1)
        return rgb.astype(np.uint8)

    def _read_mask(self, scene_dir):
        mask = self._read_tif(os.path.join(scene_dir, 'all.tif'))
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
        return mask

    def _read_tif(self, path):
        ds = gdal.Open(path)
        arr = ds.ReadAsArray()
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        scene_idx, y, x = self.patches[idx]
        img_patch = self.imgs[scene_idx][y:y+self.patch_size, x:x+self.patch_size, :]
        mask_patch = self.masks[scene_idx][y:y+self.patch_size, x:x+self.patch_size]
        
        if self.transform:
            augmented = self.transform(image=img_patch, mask=mask_patch)
            img_patch = augmented['image']  # Already torch.Tensor (C, H, W)
            mask_patch = augmented['mask']
            if not isinstance(mask_patch, torch.Tensor):
                mask_patch = torch.from_numpy(mask_patch).long()
            mask_patch = mask_patch.unsqueeze(0)
        else:
            img_patch = torch.from_numpy(img_patch).permute(2, 0, 1)
            mask_patch = torch.from_numpy(mask_patch).long()
            mask_patch = mask_patch.unsqueeze(0)
        
        return {'img': img_patch, 'mask': mask_patch, 'coords': (scene_idx, y, x)}

# ---- Boundary Loss ----
def boundary_map(mask):
    # mask: [B, 1, H, W] torch tensor, 0/1
    mask_np = mask.cpu().numpy().astype(np.uint8)
    edge_maps = []
    for m in mask_np:
        edge = cv2.Canny(m[0]*255, 50, 150)  # Canny expects 0-255 uint8
        edge_maps.append(edge / 255.)
    edge_maps = np.stack(edge_maps)
    return torch.from_numpy(edge_maps).float().unsqueeze(1).to(mask.device)

def boundary_loss(pred, target):
    pred_prob = torch.sigmoid(pred)
    pred_edge = boundary_map((pred_prob > 0.5).float())
    target_edge = boundary_map(target)
    return F.binary_cross_entropy(pred_edge, target_edge)
# ---- END Boundary Loss ----

class ComboLoss(torch.nn.Module):
    def __init__(self, dice_weight=0.6, bce_weight=0.3, boundary_weight=0.2):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, outputs, targets):
        # Убедимся, что у targets есть размерность каналов
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        loss = self.dice_weight * self.dice(outputs, targets) + \
               self.bce_weight * self.bce(outputs, targets.float())
        if self.boundary_weight > 0:
            loss += self.boundary_weight * boundary_loss(outputs, targets)
        return loss

# class ComboLoss(torch.nn.Module):
#     def __init__(self, dice_weight=0.7, bce_weight=0.3, focal_weight=0.7):
#         super().__init__()
#         self.dice = smp.losses.DiceLoss(mode='binary')
#         self.bce = torch.nn.BCEWithLogitsLoss()
#         self.dice_weight = dice_weight
#         self.bce_weight = bce_weight
#         self.focal = smp.losses.FocalLoss(mode='binary')
#         self.focal_weight = focal_weight
        
#     def forward(self, outputs, targets):
#         # Убедимся, что у targets есть размерность каналов
#         if targets.ndim == 3:
#             targets = targets.unsqueeze(1)
#         return self.dice_weight * self.dice(outputs, targets) + self.bce_weight * self.bce(outputs, targets.float()) + self.focal_weight * self.focal(outputs, targets)


def visualize_predictions(model, loader, device, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
                
            # Process one image at a time to save memory
            images = batch['img'][0:1].to(device)  # Take only one image
            masks = batch['mask'][0:1]  # Take only one image
            
            # Убедимся, что маски имеют правильную размерность
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            # Move tensors to CPU and convert to numpy immediately to free GPU memory
            img = denormalize(images[0]).cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            mask = masks[0].squeeze(0).cpu().numpy()
            pred = predictions[0].squeeze(0).cpu().numpy()
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            # Clear GPU memory
            del images, masks, outputs, predictions
            torch.cuda.empty_cache()
    
    plt.tight_layout()
    plt.show()
    plt.close()


# Пути к вашим сценам
data_dirs = ["data/ventura", "data/santa_rosa"]

# Значения для нормализации/денормализации для диапазона 0-255

val_transform = A.Compose([
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.4),
    A.CLAHE(p=0.15),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    ], p=0.15),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

patch_size = 224
stride = 32
# Создаем датасеты для train, val и test
train_dataset = SatelliteDataset(
    root_dirs=data_dirs,
    patch_size=patch_size,
    stride=stride,
    transform=train_transform,
    split='train',
    test_ratio=0.1,
    random_seed=SEED
)

val_dataset = SatelliteDataset(
    root_dirs=data_dirs,
    patch_size=patch_size,
    stride=stride,
    transform=val_transform, 
    split='val',
    test_ratio=0.1,
    random_seed=SEED
)

test_dataset = SatelliteDataset(
    root_dirs=data_dirs,
    patch_size=patch_size,
    stride=stride,
    transform=val_transform,
    split='test',
    test_ratio=0.1,
    random_seed=SEED
)

# Создаем DataLoader для каждого датасета
batch_size = 32

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Changed from 4 to 0
    pin_memory=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,  # Changed from 1 to 0
    pin_memory=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,  # Changed from 1 to 0
    pin_memory=True
)

# Функция денормализации для визуализации
def denormalize(tensor, mean=MEAN, std=STD):
    result = tensor.clone().detach()
    for t, m, s in zip(result, mean, std):
        t.mul_(s).add_(m)
    return result

# Функция для визуализации батча
def visualize_batch(loader, title):
    # Получаем один батч
    iterator = iter(loader)
    batch = next(iterator)
    
    # Распаковываем батч
    images = batch['img']
    masks = batch['mask']
    coords = batch['coords']
    scene_idx_tensor, y_tensor, x_tensor = coords
    
    # Отрисовка батча
    fig, axes = plt.subplots(min(batch_size, len(images)), 2, figsize=(10, min(batch_size, len(images)) * 5))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(batch_size, len(images))):
        # Денормализуем изображение
        img = denormalize(images[i]).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        # Маска - убираем лишние размерности и переводим в numpy
        mask = masks[i].squeeze().cpu().numpy()
        
        # Координаты
        scene_idx = scene_idx_tensor[i].item()
        y_coord = y_tensor[i].item()
        x_coord = x_tensor[i].item()
        
        # Отрисовка
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i+1}, Scene: {scene_idx}, (y={y_coord}, x={x_coord})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"Mask {i+1}")
        axes[i, 1].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Статистика
    building_pixels = (masks == 1).sum().item()
    total_pixels = masks.numel()
    building_percent = building_pixels / total_pixels * 100
    print(f"{title} - Dataset size: {len(loader.dataset)}")
    print(f"{title} - Building pixels: {building_pixels} ({building_percent:.2f}%)")
    print(f"{title} - Background pixels: {total_pixels - building_pixels} ({100 - building_percent:.2f}%)")

# Визуализация данных из всех сплитов
print("\nDataset statistics:")
print(f"Train patches: {len(train_dataset)} ({len(train_dataset)/(len(train_dataset) + len(val_dataset) + len(test_dataset))*100:.1f}%)")
print(f"Val patches: {len(val_dataset)} ({len(val_dataset)/(len(train_dataset) + len(val_dataset) + len(test_dataset))*100:.1f}%)")
print(f"Test patches: {len(test_dataset)} ({len(test_dataset)/(len(train_dataset) + len(val_dataset) + len(test_dataset))*100:.1f}%)")
print(f"Total patches: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")

# Анализ распределения патчей по сценам
scene_stats = {'train': {}, 'val': {}, 'test': {}}
for split, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
    for patch in dataset.patches:
        scene_idx = patch[0]
        scene_stats[split][scene_idx] = scene_stats[split].get(scene_idx, 0) + 1

print("\nPatch distribution by scene:")
for scene_idx in range(len(train_dataset.root_dirs)):
    print(f"\nScene {scene_idx} ({train_dataset.root_dirs[scene_idx]}):")
    total_scene_patches = sum(stats.get(scene_idx, 0) for stats in scene_stats.values())
    for split in ['train', 'val', 'test']:
        count = scene_stats[split].get(scene_idx, 0)
        percentage = count/total_scene_patches*100 if total_scene_patches > 0 else 0
        print(f"  {split}: {count} patches ({percentage:.1f}%)")

print("\nDataset sizes:")
print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")
print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")

# Отображаем по одному батчу из каждого сплита
visualize_batch(train_loader, "Training Data")
visualize_batch(val_loader, "Validation Data")
visualize_batch(test_loader, "Test Data")



# Функция потерь и метрики
criterion = ComboLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Метрики
metrics = [
    smp.metrics.iou_score,
    smp.metrics.f1_score,
    smp.metrics.precision,
    smp.metrics.recall,
]

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader):
        images = batch['img'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, metrics, device, writer, epoch):
    model.eval()
    total_loss = 0
    
    # Инициализируем статистики для метрик
    tp = torch.zeros(1, device=device)
    fp = torch.zeros(1, device=device)
    fn = torch.zeros(1, device=device)
    tn = torch.zeros(1, device=device)
    
    # Для вычисления ROC и PR на лету
    thresholds = torch.linspace(0, 1, 100, device=device)
    n_thresholds = len(thresholds)
    
    # Массивы для подсчета TP, FP, TN, FN для каждого порога
    tp_thresholds = torch.zeros(n_thresholds, device=device)
    fp_thresholds = torch.zeros(n_thresholds, device=device)
    tn_thresholds = torch.zeros(n_thresholds, device=device)
    fn_thresholds = torch.zeros(n_thresholds, device=device)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            images = batch['img'].to(device)
            masks = batch['mask'].to(device)
            
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Применяем сигмоид для получения вероятностей
            prob_mask = outputs.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
            
            # Обновляем статистику для каждого порога
            for t_idx, threshold in enumerate(thresholds):
                pred_t = (prob_mask > threshold).float()
                tp_t, fp_t, fn_t, tn_t = smp.metrics.get_stats(
                    pred_t.long(), masks.long(), mode="binary"
                )
                tp_thresholds[t_idx] += tp_t.sum()
                fp_thresholds[t_idx] += fp_t.sum()
                tn_thresholds[t_idx] += tn_t.sum()
                fn_thresholds[t_idx] += fn_t.sum()
            
            # Собираем базовую статистику для порога 0.5
            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                pred_mask.long(), masks.long(), mode="binary"
            )
            tp += batch_tp.sum()
            fp += batch_fp.sum()
            fn += batch_fn.sum()
            tn += batch_tn.sum()
            
            # Логируем изображения только для первого батча
            if i == 0:
                img = denormalize(images[0]).permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                
                gt_mask = masks[0].squeeze(0).cpu().numpy() * 255
                gt_mask = np.expand_dims(gt_mask, axis=-1)
                pred_mask_np = pred_mask[0].squeeze(0).cpu().numpy() * 255
                pred_mask_np = np.expand_dims(pred_mask_np, axis=-1)
                
                writer.add_image('Img/Images/Original', img, epoch, dataformats='HWC')
                writer.add_image('Img/Masks/Ground_Truth', gt_mask, epoch, dataformats='HWC')
                writer.add_image('Img/Masks/Prediction', pred_mask_np, epoch, dataformats='HWC')
            
            # Очищаем память
            del outputs, prob_mask, pred_mask
            torch.cuda.empty_cache()
    
    # Вычисляем ROC и PR кривые из накопленной статистики
    tpr = tp_thresholds / (tp_thresholds + fn_thresholds + 1e-7)
    fpr = fp_thresholds / (fp_thresholds + tn_thresholds + 1e-7)
    precision = tp_thresholds / (tp_thresholds + fp_thresholds + 1e-7)
    recall = tp_thresholds / (tp_thresholds + fn_thresholds + 1e-7)
    
    # Переводим в numpy для вычисления AUC
    tpr = tpr.cpu().numpy()
    fpr = fpr.cpu().numpy()
    precision = precision.cpu().numpy()
    recall = recall.cpu().numpy()
    
    # Сортируем значения для PR кривой
    sort_idx = np.argsort(recall)
    recall = recall[sort_idx]
    precision = precision[sort_idx]
    
    # Добавляем крайние точки для PR кривой
    recall = np.r_[0., recall, 1.]
    precision = np.r_[1., precision, 0.]
    
    # Интерполяция precision для монотонности (step interpolation)
    decreasing_max = np.maximum.accumulate(precision[::-1])[::-1]
    precision = decreasing_max
    
    # Вычисляем AUC для ROC и PR кривых
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)  # Используем auc для PR кривой после интерполяции
    
    # Создаем и логируем ROC кривую
    fig_roc = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    writer.add_figure('ROC_Curve', fig_roc, epoch)
    plt.close()
    
    # Создаем и логируем PR кривую
    fig_pr = plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='blue', alpha=0.8, where='post', label=f'PR curve (AP = {pr_auc:.2f})')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    writer.add_figure('PR_Curve', fig_pr, epoch)
    plt.close()
    
    # Вычисляем метрики
    metric_values = {
        'iou_score': smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item(),
        'f1_score': smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item(),
        'precision': smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item(),
        'recall': smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item(),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    return total_loss / len(loader), metric_values

num_epochs = 100
best_val_iou = 0.0  # Изменяем с loss на IoU
patience = 10
patience_counter = 0
epoch = 0

writer = SummaryWriter(log_dir=f"runs/segmentation_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_metrics = validate(model, val_loader, criterion, metrics, device, writer, epoch)
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    for metric_name, value in val_metrics.items():
        writer.add_scalar(f'Metric/val_{metric_name}', value, epoch)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    # print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss: {val_loss:.4f}')
    print('Validation Metrics:')
    for metric_name, value in val_metrics.items():
        print(f'{metric_name}: {value:.4f}')
    
    if val_metrics['iou_score'] > best_val_iou:
        best_val_iou = val_metrics['iou_score']
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
        print(f'New best IoU: {best_val_iou:.4f}')
    else:
        patience_counter += 1
        print(f'No improvement in IoU for {patience_counter} epochs')
    
    if patience_counter >= patience:
        print(f'Early stopping triggered after {epoch+1} epochs')
        break

# Загрузка лучшей модели
model.load_state_dict(torch.load('best_model.pth'))

# Тестирование на тестовом наборе
test_loss, test_metrics = validate(model, val_loader, criterion, metrics, device, writer, epoch+1)
print('\nTest Results:')
print(f'Test Loss: {test_loss:.4f}')
print('Test Metrics:')
for metric_name, value in test_metrics.items():
    print(f'{metric_name}: {value:.4f}')

# Визуализация предсказаний
visualize_predictions(model, test_loader, device)

writer.close()

