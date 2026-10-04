"""Paper dataset geometry, frozen-feature pooling and physical adapter."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import random
import cv2
import numpy as np
import matplotlib.tri as mtri
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
SEED=42
VIDEO_RANGES=[(1,30,1),(31,68,2),(69,101,3),(102,139,4),(140,171,5),(172,209,6),(210,235,7),(236,252,8),(253,272,9)]

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

def video_id(frame_file: str) -> int:
    number = int(frame_file.replace("frame_", "").replace(".jpg", ""))
    for start, end, identifier in VIDEO_RANGES:
        if start <= number <= end:
            return identifier
    raise ValueError(f"Frame fuera de los rangos de video: {frame_file}")

@dataclass
class Sample:
    frame_file: str
    image_rgb: np.ndarray
    video_id: int
    phys: np.ndarray
    target: float
    radial_cm: float
    x_norm: float
    y_norm: float

class PhysicsAdapter(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.SiLU()
        )
        self.res1 = nn.Sequential(
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.SiLU(),
            nn.Dropout(0.15), nn.Linear(256, 256), nn.BatchNorm1d(256),
        )
        self.res2 = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(),
            nn.Dropout(0.15), nn.Linear(128, 128), nn.BatchNorm1d(128),
        )
        self.downsample = nn.Linear(256, 128)
        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()
        self.head = nn.Linear(128, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_layer(features)
        hidden = self.act1(hidden + self.res1(hidden))
        hidden = self.act2(self.downsample(hidden) + self.res2(hidden))
        return self.head(hidden)

def load_samples(project_dir: Path) -> list[Sample]:
    data_dir = project_dir / "data"
    dataset_dir = data_dir / "frames"
    with (data_dir / "annotations.json").open(encoding="utf-8") as stream:
        annotations = json.load(stream)

    sim = np.load(data_dir / "fem_prior.npz")
    triangles = []
    for element in sim["elements"]:
        triangles.extend([
            [element[0], element[1], element[2]],
            [element[0], element[2], element[3]],
        ])
    mesh = mtri.Triangulation(sim["nodes"][:, 0], sim["nodes"][:, 1], triangles)
    fem = mtri.LinearTriInterpolator(mesh, sim["U_abs"])

    from .calibration import voltage as voltage_curve, decay_derivative

    samples: list[Sample] = []
    for frame_file in sorted(dataset_dir.glob("frame_*.jpg")):
        polygons = annotations.get(frame_file.name, [])
        if not polygons:
            continue
        image_bgr = cv2.imread(str(frame_file))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_bgr.shape[:2]
        conductors = []
        sphere = None
        probe = None
        for polygon in polygons:
            points = np.asarray(polygon.get("points", []), dtype=np.int32)
            if len(points) < 3:
                continue
            moments = cv2.moments(points)
            if moments["m00"] <= 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cls = polygon.get("class")
            if cls in ("esfera", "bobina_cobre"):
                conductors.append(points)
                if cls == "esfera":
                    radius = int(np.mean(np.linalg.norm(points - [cx, cy], axis=1)))
                    sphere = (cx, cy, radius)
            elif cls == "diodo_mano":
                probe = (cx, cy)
        if sphere is None:
            continue

        sx, sy, sphere_radius = sphere
        cm_per_px = 4.5 / (2.0 * sphere_radius) if sphere_radius > 0 else 0.0084
        if probe is None:
            distance_conductor_cm = np.nan
            radial_cm = np.nan
            x_norm = y_norm = np.nan
        else:
            px, py = probe
            distances = [
                -cv2.pointPolygonTest(points, (float(px), float(py)), True)
                for points in conductors
            ]
            distance_conductor_cm = max(0.0, min(distances)) * cm_per_px
            radial_cm = np.hypot(px - sx, py - sy) * cm_per_px
            x_norm = px / width
            y_norm = py / height

        if probe is None:
            v_fem = np.nan
        else:
            value_fem = fem(radial_cm / 100.0, 12.25 / 100.0)
            v_fem = 0.0 if np.ma.is_masked(value_fem) else float(value_fem)
            if np.isnan(v_fem):
                v_fem = 0.0
        target = float(voltage_curve(distance_conductor_cm))
        e_fem = float(decay_derivative(distance_conductor_cm))
        phys = np.asarray([v_fem, e_fem, radial_cm, x_norm, y_norm], dtype=np.float32)
        samples.append(Sample(
            frame_file.name, image_rgb, video_id(frame_file.name), phys,
            target, radial_cm, x_norm, y_norm,
        ))
    return samples

def make_split(samples: list[Sample], test_videos: int) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray([sample.video_id for sample in samples])
    indices = np.arange(len(samples))
    available_videos = sorted(set(groups.tolist()))
    held_out = set(available_videos[-test_videos:])
    test_idx = indices[np.isin(groups, list(held_out))]
    dev_idx = indices[~np.isin(groups, list(held_out))]
    assert set(groups[dev_idx]).isdisjoint(set(groups[test_idx]))
    return dev_idx, test_idx

def jittered_image(sample: Sample, aug_idx: int) -> np.ndarray:
    if aug_idx == 0:
        return sample.image_rgb
    # Solo fotometría: rotar/voltear sin transformar la coordenada ROI desalineaba
    # el parche visual respecto de la sonda en los scripts anteriores.
    torch.manual_seed(SEED + aug_idx * 1009 + int(sample.frame_file[6:9]))
    jitter = transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03)
    return np.asarray(jitter(transforms.ToPILImage()(sample.image_rgb)))

def weighted_local(tokens: torch.Tensor, samples: list[Sample], indices: list[int], grid: int) -> np.ndarray:
    weights = np.asarray([[0.05, 0.10, 0.05], [0.10, 0.40, 0.10], [0.05, 0.10, 0.05]])
    output = []
    token_np = tokens.detach().cpu().numpy()
    for row, sample_idx in enumerate(indices):
        sample = samples[sample_idx]
        center_x = int(np.clip((sample.x_norm if np.isfinite(sample.x_norm) else .5) * grid, 0, grid - 1))
        center_y = int(np.clip((sample.y_norm if np.isfinite(sample.y_norm) else .5) * grid, 0, grid - 1))
        local = np.zeros(token_np.shape[-1], dtype=np.float32)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                x = int(np.clip(center_x + dx, 0, grid - 1))
                y = int(np.clip(center_y + dy, 0, grid - 1))
                local += weights[dy + 1, dx + 1] * token_np[row, y * grid + x]
        output.append(local)
    return np.asarray(output)

def expanded(indices: np.ndarray, clean: np.ndarray, augmented: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    variants = np.concatenate([clean[indices, None, :], augmented[indices]], axis=1)
    x = variants.reshape(-1, variants.shape[-1])
    y = np.repeat(values[indices], variants.shape[1], axis=0)
    return x, y

def train_adapter(x: np.ndarray, y: np.ndarray, input_dim: int, epochs: int,
                  device: torch.device, seed: int) -> PhysicsAdapter:
    seed_everything(seed)
    model = PhysicsAdapter(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    ), batch_size=32, shuffle=True, generator=generator)
    criterion = nn.SmoothL1Loss(beta=0.5)
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model.eval()

def predict(model: PhysicsAdapter, x: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        return model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy().ravel()
