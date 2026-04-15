# train_mlp_baseline.py

"""
Baseline node classification (no graph) across 12 Brazilian cities.
Structured for notebook use, with callable functions:
- run_within_city()
- run_cross_city()
- run_leave_one_out()
Each uses a simple balanced MLP classifier.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


# ================================================================
# 0) PARAMETERS
# ================================================================
DATA_DIR = "./data/dataloaders"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 512
LR = 1e-4
WEIGHT_DECAY = 5e-4
SEED = 42
HIDDEN_DIM = 128
GRAPH_VERSION = "full"  # ["full", "geometry_only", "google_only"]
NUM_HOPS = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

CITY_LIST = [
    "São Paulo, Brazil",
    "Rio de Janeiro, Brazil",
    "Federal District, Brazil",
    "Salvador, Brazil",
    "Fortaleza, Brazil",
    "Belo Horizonte, Brazil",
    "Manaus, Brazil",
    "Curitiba, Brazil",
    "Recife, Brazil",
    "Goiânia, Brazil",
    "Belém, Brazil",
    "Porto Alegre, Brazil",
]


# ================================================================
# 1) DATA UTILITIES
# ================================================================
def undersample_balanced(x, y, rng):
    """Random undersampling to balance classes."""
    y_np = y.numpy()
    idx0 = np.where(y_np == 0)[0]
    idx1 = np.where(y_np == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return x, y
    n = min(len(idx0), len(idx1))
    idx0_sel = rng.choice(idx0, n, replace=False)
    idx1_sel = rng.choice(idx1, n, replace=False)
    idx_sel = np.concatenate([idx0_sel, idx1_sel])
    rng.shuffle(idx_sel)
    return x[idx_sel], y[idx_sel]


def load_city_data(city_name, data_dir, version, num_hops):
    """Load x, y tensors from dataloader file."""
    safe_name = city_name.lower().replace(",", "").replace(" ", "_")
    path = os.path.join(
        data_dir, f"dataloader_{safe_name}_{version}_h{num_hops}.pt"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file for {city_name}: {path}")

    graphs = torch.load(path, map_location="cpu", weights_only=False)
    x_all = torch.cat([g.x[0].unsqueeze(0) for g in graphs])
    y_all = torch.cat([g.y for g in graphs])
    return x_all, y_all


def load_all_cities(data_dir, version, num_hops, city_list):
    """Preload data for all cities into a dict."""
    city_data = {}
    for city in city_list:
        print(f"Loading {city}...")
        city_data[city] = load_city_data(city, data_dir, version, num_hops)
    return city_data


# ================================================================
# 2) MODEL
# ================================================================
class MLPBaseline(nn.Module):
    """Simple MLP for node classification."""

    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ================================================================
# 3) TRAIN / EVAL
# ================================================================
def train_epoch(loader, model, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


def eval_epoch(loader, model, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    acc = (preds == targets).mean()
    f1 = f1_score(targets, preds, zero_division=0)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    return acc, f1, prec, rec


def run_experiment(train_x, train_y, test_x, test_y):
    """Train MLP and evaluate."""
    in_dim = train_x.size(1)
    num_classes = int(train_y.max().item() + 1)
    model = MLPBaseline(in_dim, HIDDEN_DIM, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        list(zip(train_x, train_y)), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        list(zip(test_x, test_y)), batch_size=BATCH_SIZE, shuffle=False
    )

    for _ in range(EPOCHS):
        train_epoch(train_loader, model, optimizer, criterion, DEVICE)
    return eval_epoch(test_loader, model, DEVICE)


# ================================================================
# 4) EXPERIMENT MODES
# ================================================================
def run_within_city(city_data):
    """Train/test on the same city (80/20 split, balanced)."""
    rng = np.random.default_rng(SEED)
    results = {}
    print("\n=== WITHIN-CITY ===")
    for city, (x, y) in city_data.items():
        x, y = undersample_balanced(x, y, rng)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        split = int(0.8 * len(y))
        train_x, test_x = x[idx[:split]], x[idx[split:]]
        train_y, test_y = y[idx[:split]], y[idx[split:]]
        acc, f1, prec, rec = run_experiment(train_x, train_y, test_x, test_y)
        results[city] = (acc, f1, prec, rec)
        print(f"{city:25s} | Acc {acc:.4f} | F1 {f1:.4f}")
    return results


def run_cross_city(city_data):
    """Train on one city, test on all others."""
    rng = np.random.default_rng(SEED)
    results = {}
    print("\n=== CROSS-CITY ===")
    for train_city, (x_train, y_train) in city_data.items():
        x_train, y_train = undersample_balanced(x_train, y_train, rng)
        results[train_city] = {}
        for test_city, (x_test, y_test) in city_data.items():
            acc, f1, prec, rec = run_experiment(x_train, y_train, x_test, y_test)
            results[train_city][test_city] = (acc, f1, prec, rec)
            print(f"Train {train_city[:15]:15s} → Test {test_city[:15]:15s} | F1 {f1:.4f}")
    return results


def run_leave_one_out(city_data):
    """Train on 11 cities, test on the remaining one."""
    rng = np.random.default_rng(SEED)
    results = {}
    print("\n=== LEAVE-ONE-OUT ===")
    for test_city in city_data.keys():
        x_train_all, y_train_all = [], []
        for city, (x, y) in city_data.items():
            if city == test_city:
                continue
            x, y = undersample_balanced(x, y, rng)
            x_train_all.append(x)
            y_train_all.append(y)
        x_train = torch.cat(x_train_all)
        y_train = torch.cat(y_train_all)
        x_test, y_test = city_data[test_city]
        acc, f1, prec, rec = run_experiment(x_train, y_train, x_test, y_test)
        results[test_city] = (acc, f1, prec, rec)
        print(f"Leave-out {test_city:25s} | Acc {acc:.4f} | F1 {f1:.4f}")
    return results


city_data = load_all_cities(DATA_DIR, GRAPH_VERSION, NUM_HOPS, CITY_LIST)
results_within = run_within_city(city_data)
results_cross = run_cross_city(city_data)
results_loo = run_leave_one_out(city_data)
