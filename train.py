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
import random
import cv2
from torch.amp import GradScaler, autocast

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


########################################################
# MODEL
# customize model with PVTv2 encoder
########################################################
class PVTV2Encoder(torch.nn.Module, EncoderMixin):
    """PVTv2 encoder for semantic segmentation.

    This class implements an encoder based on the PVTv2 (Pyramid Vision Transformer v2) architecture,
    specifically designed for semantic segmentation tasks. It wraps the PVTv2 model from the timm library
    and adapts it to work with the segmentation_models_pytorch framework.

    The encoder extracts multi-scale features from the input image using the PVTv2 backbone,
    producing feature maps at different spatial resolutions that can be used by various decoder architectures.

    Attributes:
        _out_channels (List[int]): Number of channels in each output feature map
        _depth (int): Number of downsampling operations in the encoder
        _in_channels (int): Number of input channels (3 for RGB images)
        _output_stride (int): Total downsampling factor of the encoder

    Properties:
        output_stride: Returns the total downsampling factor of the encoder

    Methods:
        forward(x): Processes input tensor and returns list of feature maps
        load_state_dict(): Loads pretrained weights into the model

    Based on https://smp.readthedocs.io/en/latest/insights.html#creating-your-own-encoder
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model('pvt_v2_b5', pretrained=True, features_only=True)
        channels = self.model.feature_info.channels()
        self._out_channels = channels  # [64, 128, 320, 512]
        self._depth = 4
        self._in_channels = 3
        self._output_stride = 32

    @property
    def output_stride(self):
        return self._output_stride

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.model(x)
        return features

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.model.load_state_dict(state_dict, *args, **kwargs)

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

########################################################
# MODEL
# Tiny baseline model example
########################################################

# model = smp.UnetPlusPlus(
#     encoder_name="efficientnet-b2", # mobilenet_v2
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=1
# )


model = model.to(DEVICE)


########################################################
# DATASET
########################################################
class SatelliteDataset(Dataset):
    """A PyTorch Dataset for loading satellite imagery and segmentation masks.

    This dataset handles loading and preprocessing of satellite imagery and corresponding
    segmentation masks. It splits scenes into patches using a grid-based approach and 
    supports train/val/test splits.

    Args:
        root_dirs (Union[str, List[str]]): Path(s) to scene folder(s) containing imagery and masks
        patch_size (int, optional): Size of extracted patches. Defaults to 512.
        stride (int, optional): Stride between patches. Defaults to 512.
        transform (callable, optional): Transforms to apply to patches. Defaults to None.
        split (str, optional): Dataset split - 'train', 'val' or 'test'. Defaults to 'train'.
        test_ratio (float, optional): Ratio of data to use for test set. Defaults to 0.1.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    def _calculate_cell_building_density(self, mask, cell_size_h, cell_size_w, grid_h, grid_w):
        """Вычисляет плотность зданий для каждой ячейки сетки."""
        densities = {}
        for i in range(grid_h):
            for j in range(grid_w):
                # Определяем границы ячейки
                y_start = i * cell_size_h
                y_end = min((i + 1) * cell_size_h, mask.shape[0])
                x_start = j * cell_size_w
                x_end = min((j + 1) * cell_size_w, mask.shape[1])
                
                # Вычисляем плотность зданий в ячейке
                cell_mask = mask[y_start:y_end, x_start:x_end]
                building_pixels = np.sum(cell_mask > 0)
                total_pixels = cell_mask.size
                density = building_pixels / total_pixels if total_pixels > 0 else 0
                densities[(i, j)] = density
                
                # Отладочный вывод для проверки плотности
                if building_pixels > 0:
                    print(f"Cell ({i},{j}) has density {density:.3f} ({building_pixels}/{total_pixels} pixels)")
        
        return densities

    def _pad_patch(self, img, mask, y, x, patch_size):
        """Pad patch with zeros if it extends beyond image boundaries.
        
        Args:
            img (np.ndarray): Source image
            mask (np.ndarray): Source mask
            y (int): Y coordinate of patch top-left corner
            x (int): X coordinate of patch top-left corner
            patch_size (int): Size of the patch
            
        Returns:
            tuple: (padded_img, padded_mask, is_valid)
        """
        h, w = img.shape[:2]
        
        # Calculate actual patch dimensions
        actual_h = min(patch_size, h - y)
        actual_w = min(patch_size, w - x)
        
        # Create empty patches
        if len(img.shape) == 3:
            padded_img = np.zeros((patch_size, patch_size, img.shape[2]), dtype=img.dtype)
        else:
            padded_img = np.zeros((patch_size, patch_size), dtype=img.dtype)
        padded_mask = np.zeros((patch_size, patch_size), dtype=mask.dtype)
        
        # Copy actual data
        padded_img[:actual_h, :actual_w] = img[y:y+actual_h, x:x+actual_w]
        padded_mask[:actual_h, :actual_w] = mask[y:y+actual_h, x:x+actual_w]
        
        # Check if patch contains any buildings
        has_buildings = np.any(padded_mask > 0)
        
        return padded_img, padded_mask, has_buildings

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
        for scene_idx, (img, mask) in enumerate(zip(self.imgs, self.masks)):
            print(f"\nProcessing scene {scene_idx}:")
            h, w = img.shape[:2]
            
            # Делаем размер ячейки в 3-4 раза больше patch_size, кратным stride
            min_cell_size = 3 * patch_size
            cell_size = ((min_cell_size + stride - 1) // stride) * stride
            
            # Вычисляем количество ячеек сетки
            grid_h = max(2, (h - 1) // cell_size + 1)  # Changed to include partial cells
            grid_w = max(2, (w - 1) // cell_size + 1)  # Changed to include partial cells
            
            # Корректируем размер ячейки
            cell_h = (h - 1) // grid_h + 1  # Changed to include partial cells
            cell_w = (w - 1) // grid_w + 1  # Changed to include partial cells
            cell_size_h = ((cell_h + stride - 1) // stride) * stride
            cell_size_w = ((cell_w + stride - 1) // stride) * stride
            
            print(f"Grid size: {grid_h}x{grid_w}, Cell size: {cell_size_h}x{cell_size_w}")
            
            # Вычисляем плотность зданий для каждой ячейки
            densities = self._calculate_cell_building_density(mask, cell_size_h, cell_size_w, grid_h, grid_w)
            
            # Находим максимальную плотность для нормализации
            max_density = max(densities.values()) if densities else 0
            
            # Разделяем ячейки на группы по плотности
            cells_by_density = {}
            for cell, density in densities.items():
                # Нормализуем плотность и разбиваем на 4 группы
                if max_density > 0:
                    normalized_density = density / max_density
                    group = min(3, int(normalized_density * 4))  # 0: нет/очень мало, 1: мало, 2: средне, 3: много
                else:
                    group = 0
                if group not in cells_by_density:
                    cells_by_density[group] = []
                cells_by_density[group].append(cell)
            
            # Выводим статистику по группам
            print("\nCell distribution by density groups:")
            for group in sorted(cells_by_density.keys()):
                density_range = [f"{g/4:.1f}-{(g+1)/4:.1f}" for g in [group]][0]
                print(f"Group {group} (density {density_range}): {len(cells_by_density[group])} cells")
            
            # Выбираем тестовые ячейки стратифицированно
            test_cells = set()
            n_test_cells = max(2, int(grid_h * grid_w * self.test_ratio))
            print(f"\nSelecting {n_test_cells} test cells")
            
            total_cells = sum(len(cells) for cells in cells_by_density.values())
            remaining_cells = n_test_cells
            
            # Сначала выбираем по одной ячейке из каждой непустой группы
            for group in sorted(cells_by_density.keys(), reverse=True):  # Начинаем с групп с большей плотностью
                cells = cells_by_density[group]
                if not cells or remaining_cells <= 0:
                    continue
                
                # Берем одну ячейку из группы
                selected_idx = np.random.choice(len(cells))
                test_cells.add(cells[selected_idx])
                cells.pop(selected_idx)  # Удаляем выбранную ячейку
                remaining_cells -= 1
            
            # Распределяем оставшиеся ячейки пропорционально размеру групп
            if remaining_cells > 0:
                total_remaining = sum(len(cells) for cells in cells_by_density.values())
                for group in sorted(cells_by_density.keys(), reverse=True):
                    cells = cells_by_density[group]
                    if not cells or remaining_cells <= 0:
                        continue
                    
                    n_group_test = max(1, int(len(cells) * remaining_cells / total_remaining))
                    n_group_test = min(n_group_test, remaining_cells, len(cells))
                    
                    if n_group_test > 0:
                        selected_idx = np.random.choice(len(cells), size=n_group_test, replace=False)
                        test_cells.update(cells[i] for i in selected_idx)
                        remaining_cells -= n_group_test
            
            print(f"\nTotal selected test cells: {len(test_cells)}")
            
            # Generate patches
            patches_with_buildings = 0
            total_patches = 0
            
            # Изменено: теперь проходим по всему изображению, включая края
            for y in range(0, h, self.stride):
                for x in range(0, w, self.stride):
                    # Get padded patches to check for buildings
                    _, _, has_buildings = self._pad_patch(img, mask, y, x, self.patch_size)
                    
                    # Calculate which cell this patch belongs to
                    patch_corners = [
                        (y // cell_size_h, x // cell_size_w),
                        (y // cell_size_h, min((x + patch_size - 1), w - 1) // cell_size_w),
                        (min((y + patch_size - 1), h - 1) // cell_size_h, x // cell_size_w),
                        (min((y + patch_size - 1), h - 1) // cell_size_h, min((x + patch_size - 1), w - 1) // cell_size_w)
                    ]
                    
                    corners_in_test = sum(1 for corner in patch_corners if corner in test_cells)
                    
                    if split == 'test':
                        if corners_in_test == len(patch_corners):
                            self.patches.append((scene_idx, y, x))
                            total_patches += 1
                            if has_buildings:
                                patches_with_buildings += 1
                    else:  # train/val
                        if corners_in_test == 0:
                            self.patches.append((scene_idx, y, x))
                            total_patches += 1
                            if has_buildings:
                                patches_with_buildings += 1
            
            print(f"\nPatches statistics for {split} split:")
            print(f"Total patches: {total_patches}")
            print(f"Patches with buildings: {patches_with_buildings} ({patches_with_buildings/total_patches*100:.2f}%)")

        # For train/val split
        if split != 'test':
            np.random.shuffle(self.patches)
            n_total = len(self.patches)
            n_train = int(n_total * 0.8)
            if split == 'train':
                self.patches = self.patches[:n_train]
            else:  # val
                self.patches = self.patches[n_train:]

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
        
        # Get padded patches
        img_patch, mask_patch, _ = self._pad_patch(
            self.imgs[scene_idx], 
            self.masks[scene_idx],
            y, x, 
            self.patch_size
        )
        
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

########################################################
# LOSS
########################################################

class ConditionalBoundaryLoss(torch.nn.Module):
    """Boundary-aware loss function for semantic segmentation.

    This loss function computes a weighted boundary loss that focuses on the edges
    of segmented regions. It uses morphological operations to detect boundaries and
    applies higher weights to boundary pixels.

    Args:
        theta (int, optional): Boundary width parameter. Defaults to 3.
        window_size (int, optional): Size of window for morphological operations. Defaults to 3.

    Attributes:
        theta (int): Boundary width parameter
        window_size (int): Size of window for morphological operations
    """

    def __init__(self, theta=3, window_size=3):
        super().__init__()
        self.theta = theta
        self.window_size = window_size
        self.pool = torch.nn.AvgPool2d(window_size, stride=1, padding=window_size//2)

    def _boundary_map(self, mask):
        """Compute boundary map from binary segmentation mask.

        This function calculates a boundary map by applying morphological operations
        to detect edges in the binary segmentation mask.

        Args:
            mask (torch.Tensor): Binary segmentation mask tensor

        Returns:
            torch.Tensor: Boundary map tensor with same shape as input mask
        """
        # Переводим в numpy и расширяем диапазон до 0-255
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        
        edge_maps = []
        for m in mask_np:
            # Применяем размытие перед детектором границ для лучшего результата
            blurred = cv2.GaussianBlur(m[0], (3, 3), 0)
            # Используем более низкие пороги для Canny, так как у нас бинарная маска
            edge = cv2.Canny(blurred, 30, 100)
            # Нормализуем обратно к 0-1
            edge_maps.append(edge / 255.)
        
        edge_maps = np.stack(edge_maps)
        return torch.from_numpy(edge_maps).float().unsqueeze(1).to(mask.device)
    
    def forward(self, pred, target):
        """Compute the boundary-aware loss.

        Args:
            pred (torch.Tensor): Predicted segmentation map
            target (torch.Tensor): Ground truth segmentation map

        Returns:
            torch.Tensor: Computed boundary loss value
        """
        # Получаем вероятности
        pred_prob = torch.sigmoid(pred)
        
        # Получаем границы из целевой маски
        target_edges = self._boundary_map(target)
        
        # Вычисляем локальную неопределенность
        local_uncertainty = -pred_prob * torch.log(pred_prob + 1e-6) - \
                          (1 - pred_prob) * torch.log(1 - pred_prob + 1e-6)
        uncertainty_map = self.pool(local_uncertainty)
        
        # Вычисляем веса для каждого пикселя на основе неопределенности
        weights = torch.exp(-uncertainty_map / self.theta)
        
        # Вычисляем градиенты предсказаний
        grad_y = torch.abs(pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :])
        grad_x = torch.abs(pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1])
        
        # Обрезаем веса и target_edges под размер градиентов
        weights_y = weights[:, :, 1:, :]
        weights_x = weights[:, :, :, 1:]
        target_edges_y = target_edges[:, :, 1:, :]
        target_edges_x = target_edges[:, :, :, 1:]
        
        # Вычисляем взвешенные потери отдельно для границ и не-границ
        # Для границ: поощряем высокие градиенты
        boundary_loss_y = (1 - grad_y) * target_edges_y * weights_y
        boundary_loss_x = (1 - grad_x) * target_edges_x * weights_x
        
        # Для не-границ: штрафуем высокие градиенты
        smoothness_loss_y = grad_y * (1 - target_edges_y) * weights_y
        smoothness_loss_x = grad_x * (1 - target_edges_x) * weights_x
        
        # Комбинируем потери
        total_loss = (boundary_loss_y.mean() + boundary_loss_x.mean() + 
                     smoothness_loss_y.mean() + smoothness_loss_x.mean()) / 4.0
        
        return total_loss

class ComboLoss(torch.nn.Module):
    """Combined loss function for semantic segmentation.

    This loss function combines multiple loss terms including Dice loss,
    Binary Cross Entropy (BCE) loss, and boundary loss with configurable weights.

    Args:
        dice_weight (float, optional): Weight for Dice loss term. Defaults to 0.6.
        bce_weight (float, optional): Weight for BCE loss term. Defaults to 0.3.
        boundary_weight (float, optional): Weight for boundary loss term. Defaults to 0.2.

    Attributes:
        dice_weight (float): Weight for Dice loss term
        bce_weight (float): Weight for BCE loss term
        boundary_weight (float): Weight for boundary loss term
        boundary_loss (ConditionalBoundaryLoss): Boundary loss function instance
    """

    def __init__(self, dice_weight=0.6, bce_weight=0.3, boundary_weight=0.2):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.boundary = ConditionalBoundaryLoss()
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, outputs, targets):
        """Compute the combined loss.

        Args:
            outputs (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Computed combined loss value
        """
        # Убедимся, что у targets есть размерность каналов
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        loss = self.dice_weight * self.dice(outputs, targets) + \
               self.bce_weight * self.bce(outputs, targets.float())
        if self.boundary_weight > 0:
            loss += self.boundary_weight * self.boundary(outputs, targets)
        return loss


def visualize_predictions(model, loader, device, num_samples=4):
    """Visualize model predictions on sample images.

    This function generates and displays side-by-side comparisons of input images,
    ground truth masks, and model predictions for a specified number of samples.

    Args:
        model (torch.nn.Module): Trained model for inference
        loader (DataLoader): DataLoader containing validation/test data
        device (torch.device): Device to run inference on
        num_samples (int, optional): Number of samples to visualize. Defaults to 4.

    Returns:
        None
    """
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
                
            images = batch['img'][0:1].to(device)  # Take only one image
            masks = batch['mask'][0:1]  # Take only one image
            
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


########################################################
# DATA
########################################################

data_dirs = ["data/ventura", "data/santa_rosa"]

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

batch_size = 64
patch_size = 224
stride = 32

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

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Denorm for visualization
def denormalize(tensor, mean=MEAN, std=STD):
    """Denormalize an image tensor to original scale.

    Args:
        tensor (torch.Tensor): Normalized image tensor
        mean (tuple, optional): Mean values used for normalization. Defaults to MEAN.
        std (tuple, optional): Standard deviation values used for normalization. Defaults to STD.

    Returns:
        torch.Tensor: Denormalized image tensor
    """
    result = tensor.clone().detach()
    for t, m, s in zip(result, mean, std):
        t.mul_(s).add_(m)
    return result

def visualize_splits_overlap(train_dataset, val_dataset, test_dataset, scene_idx=0):
    """Visualize patches from all splits on one image to check for overlaps.
    
    Args:
        train_dataset (SatelliteDataset): Training dataset
        val_dataset (SatelliteDataset): Validation dataset
        test_dataset (SatelliteDataset): Test dataset
        scene_idx (int): Index of scene to visualize
    """
    # Get original image and mask
    img = train_dataset.imgs[scene_idx]
    mask = train_dataset.masks[scene_idx]
    patch_size = train_dataset.patch_size
    
    # Create visualization
    plt.figure(figsize=(20, 10))
    
    # Plot original image with patches
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Patch Distribution')
    
    # Draw grid
    h, w = img.shape[:2]
    min_cell_size = 3 * patch_size
    cell_size = ((min_cell_size + train_dataset.stride - 1) // train_dataset.stride) * train_dataset.stride
    grid_h = max(2, (h - patch_size) // cell_size + 1)
    grid_w = max(2, (w - patch_size) // cell_size + 1)
    cell_h = (h - patch_size) // grid_h + 1
    cell_w = (w - patch_size) // grid_w + 1
    cell_size_h = ((cell_h + train_dataset.stride - 1) // train_dataset.stride) * train_dataset.stride
    cell_size_w = ((cell_w + train_dataset.stride - 1) // train_dataset.stride) * train_dataset.stride
    
    # Draw grid lines
    for i in range(grid_h + 1):
        y = i * cell_size_h
        plt.axhline(y=y, color='white', linestyle='--', alpha=0.3)
    for j in range(grid_w + 1):
        x = j * cell_size_w
        plt.axvline(x=x, color='white', linestyle='--', alpha=0.3)
    
    # Draw patches for each split with different colors and styles
    colors = {
        'train': ('blue', 0.3),
        'val': ('red', 0.3),
        'test': ('yellow', 0.3)
    }
    
    # Draw test patches first (they will be at the bottom)
    for scene_i, y, x in test_dataset.patches:
        if scene_i == scene_idx:
            rect = patches.Rectangle(
                (x, y), patch_size, patch_size,
                linewidth=2, edgecolor=colors['test'][0], 
                facecolor='none',
                alpha=1.0
            )
            plt.gca().add_patch(rect)
    
    # Then validation patches
    for scene_i, y, x in val_dataset.patches:
        if scene_i == scene_idx:
            rect = patches.Rectangle(
                (x, y), patch_size, patch_size,
                linewidth=2, edgecolor=colors['val'][0], 
                facecolor='none',
                alpha=1.0
            )
            plt.gca().add_patch(rect)
    
    # Finally train patches (they will be on top)
    for scene_i, y, x in train_dataset.patches:
        if scene_i == scene_idx:
            rect = patches.Rectangle(
                (x, y), patch_size, patch_size,
                linewidth=2, edgecolor=colors['train'][0], 
                facecolor='none',
                alpha=1.0
            )
            plt.gca().add_patch(rect)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor='blue', label='Train'),
        patches.Patch(facecolor='none', edgecolor='red', label='Val'),
        patches.Patch(facecolor='none', edgecolor='yellow', label='Test')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('off')
    
    # Plot building mask with grid
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Building Mask with Grid')
    
    # Draw grid lines
    for i in range(grid_h + 1):
        y = i * cell_size_h
        plt.axhline(y=y, color='red', linestyle='--', alpha=0.3)
    for j in range(grid_w + 1):
        x = j * cell_size_w
        plt.axvline(x=x, color='red', linestyle='--', alpha=0.3)
    plt.axis('off')
    
    plt.suptitle(f'Scene {scene_idx} - Split Distribution and Building Mask', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nScene {scene_idx} Statistics:")
    train_patches = sum(1 for p in train_dataset.patches if p[0] == scene_idx)
    val_patches = sum(1 for p in val_dataset.patches if p[0] == scene_idx)
    test_patches = sum(1 for p in test_dataset.patches if p[0] == scene_idx)
    total_patches = train_patches + val_patches + test_patches
    
    print(f"Train patches: {train_patches} ({train_patches/total_patches*100:.1f}%)")
    print(f"Val patches: {val_patches} ({val_patches/total_patches*100:.1f}%)")
    print(f"Test patches: {test_patches} ({test_patches/total_patches*100:.1f}%)")
    print(f"Total patches: {total_patches}")

# Analyze each scene
print("\nChecking for overlaps between splits...")
for scene_idx in range(len(train_dataset.root_dirs)):
    visualize_splits_overlap(train_dataset, val_dataset, test_dataset, scene_idx)


# Функция для визуализации батча
def visualize_batch(loader, title):
    """Visualize a batch of training data.

    This function displays a grid of images and their corresponding masks
    from a single batch of the data loader.

    Args:
        loader (DataLoader): DataLoader containing the dataset
        title (str): Title for the visualization plot

    Returns:
        None
    """
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

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    """Train the model for one epoch.

    This function performs one training epoch, iterating over all batches
    in the data loader and updating model parameters using mixed precision training.

    Args:
        model (torch.nn.Module): Model to train
        loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to run training on
        scaler (GradScaler): Gradient scaler for mixed precision training

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader):
        images = batch['img'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Automatic mixed precision
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # Scale loss and call backward()
        scaler.scale(loss).backward()
        
        # Unscale gradients and call/skip optimizer.step()
        scaler.step(optimizer)
        
        # Update scaler for next iteration
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, metrics, device, writer, epoch):
    """Validate the model on validation dataset.

    This function evaluates the model performance on the validation set,
    computing various metrics and logging results to TensorBoard.
    Uses mixed precision for faster evaluation.

    Args:
        model (torch.nn.Module): Model to evaluate
        loader (DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        metrics (dict): Dictionary of metric functions to compute
        device (torch.device): Device to run validation on
        writer (SummaryWriter): TensorBoard writer instance
        epoch (int): Current epoch number

    Returns:
        tuple: Average validation loss and metrics dictionary
    """
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
    
    with torch.no_grad(), autocast('cuda'):
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

num_epochs = 1000
best_val_iou = 0.0  # Изменяем с loss на IoU
patience = 10
patience_counter = 0
epoch = 0

exp_name = f"runs/segmentation_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=exp_name)

# Initialize gradient scaler for mixed precision training
scaler = GradScaler()

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
    val_loss, val_metrics = validate(model, val_loader, criterion, metrics, DEVICE, writer, epoch)
    
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
        torch.save(model.state_dict(), f"{exp_name}/best_model.pth")
        patience_counter = 0
        print(f'New best IoU: {best_val_iou:.4f}')
    else:
        patience_counter += 1
        print(f'No improvement in IoU for {patience_counter} epochs')
    
    if patience_counter >= patience:
        print(f'Early stopping triggered after {epoch+1} epochs')
        break

# Загрузка лучшей модели
model.load_state_dict(torch.load(f"{exp_name}/best_model.pth"))

# Тестирование на тестовом наборе
test_loss, test_metrics = validate(model, val_loader, criterion, metrics, DEVICE, writer, epoch+1)
print('\nTest Results:')
print(f'Test Loss: {test_loss:.4f}')
print('Test Metrics:')
for metric_name, value in test_metrics.items():
    print(f'{metric_name}: {value:.4f}')

# Визуализация предсказаний
visualize_predictions(model, test_loader, DEVICE)

writer.close()
