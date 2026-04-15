# train_gcn.py

"""
EgoGCN experiments on multiple Brazilian cities:
 - Within-city (80/20 split)
 - Leave-One-Out (train on 11 cities, test on 1)
 - Cross-City (train on one city, test on all others)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv
from sklearn.metrics import f1_score, precision_score, recall_score

# ================================================================
# PARAMETERS
# ================================================================
DATA_DIR = "./data/dataloaders"
CITY_LIST = [
    "são_paulo_brazil",
    "rio_de_janeiro_brazil",
    "federal_district_brazil",
    "salvador_brazil",
    "fortaleza_brazil",
    "belo_horizonte_brazil",
    "manaus_brazil",
    "curitiba_brazil",
    "recife_brazil",
    "goiânia_brazil",
    "belém_brazil",
    "porto_alegre_brazil",
]

BATCH_SIZE = 320
HIDDEN_DIM = 124
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 5e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(SEED)


# ================================================================
# UTILITIES
# ================================================================
def load_city_graphs(city_name, data_dir):
    """Load the precomputed local graphs for a given city."""
    safe_name = city_name.lower().replace(",", "").replace(" ", "_")
    path = os.path.join(data_dir, f"dataloader_{safe_name}_google_only_h1.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No dataloader found for {city_name} -> {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def undersample_graphs(graph_list, rng):
    """Random undersampling to balance classes."""
    labels = np.array([int(g.y.item()) for g in graph_list])
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return graph_list
    n = min(len(idx0), len(idx1))
    idx0_sel = rng.choice(idx0, n, replace=False)
    idx1_sel = rng.choice(idx1, n, replace=False)
    balanced_idx = np.concatenate([idx0_sel, idx1_sel])
    rng.shuffle(balanced_idx)
    return [graph_list[i] for i in balanced_idx]


# ================================================================
# MODEL AND TRAINING
# ================================================================
class EgoGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, edge_dim):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, in_dim * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_dim * hidden_dim, in_dim * hidden_dim),
        )
        self.conv1 = NNConv(in_channels=in_dim, out_channels=hidden_dim, nn=nn1, aggr="mean")

        nn2 = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_dim * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim),
        )
        self.conv2 = NNConv(in_channels=hidden_dim, out_channels=hidden_dim, nn=nn2, aggr="mean")

        self.cls = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        centers = batch.ptr[:-1]
        center_repr = x[centers]
        return self.cls(center_repr)


def train_epoch(loader, model, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / max(1, n)


def eval_epoch(loader, model, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1).cpu().numpy()
            tgt = batch.y.view(-1).cpu().numpy()
            preds.append(pred)
            targets.append(tgt)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    acc = (preds == targets).mean()
    f1 = f1_score(targets, preds, zero_division=0)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    return acc, f1, prec, rec


# ================================================================
# 1) WITHIN-CITY
# ================================================================
def run_within_city(city_name):
    print(f"\n=== WITHIN-CITY: {city_name} ===")
    graphs = load_city_graphs(city_name, DATA_DIR)
    indices = np.arange(len(graphs))
    rng.shuffle(indices)
    split = int(0.8 * len(graphs))
    train_idx, test_idx = indices[:split], indices[split:]
    train_graphs = undersample_graphs([graphs[i] for i in train_idx], rng)
    test_graphs = undersample_graphs([graphs[i] for i in test_idx], rng)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    sample = graphs[0]
    in_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)
    num_classes = int(torch.cat([g.y for g in graphs]).max().item() + 1)

    model = EgoGCN(in_dim, HIDDEN_DIM, num_classes, edge_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(train_loader, model, optimizer, criterion, DEVICE)
        acc, f1, prec, rec = eval_epoch(test_loader, model, DEVICE)
        print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Acc {acc:.4f} | F1 {f1:.4f} | Prec {prec:.4f} | Rec {rec:.4f}")


# ================================================================
# 2) LEAVE-ONE-OUT
# ================================================================
def run_leave_one_out():
    print("\n=== LEAVE-ONE-OUT (train on 11, test on 1) ===")
    for test_city in CITY_LIST:
        print(f"\n--- Test city: {test_city} ---")
        train_graphs, test_graphs = [], []
        for city in CITY_LIST:
            graphs = load_city_graphs(city, DATA_DIR)
            if city == test_city:
                test_graphs.extend(graphs)
            else:
                train_graphs.extend(graphs)
        train_graphs = undersample_graphs(train_graphs, rng)
        test_graphs = undersample_graphs(test_graphs, rng)

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)
        sample = train_graphs[0]
        in_dim = sample.x.size(1)
        edge_dim = sample.edge_attr.size(1)
        num_classes = int(torch.cat([g.y for g in train_graphs]).max().item() + 1)

        model = EgoGCN(in_dim, HIDDEN_DIM, num_classes, edge_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(train_loader, model, optimizer, criterion, DEVICE)
            acc, f1, prec, rec = eval_epoch(test_loader, model, DEVICE)
            print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Acc {acc:.4f} | F1 {f1:.4f} | Prec {prec:.4f} | Rec {rec:.4f}")


# ================================================================
# 3) CROSS-CITY (train one, test all others)
# ================================================================
def run_cross_city():
    print("\n=== CROSS-CITY (train one city, test all others) ===")
    results = np.zeros((len(CITY_LIST), len(CITY_LIST)))
    for i, train_city in enumerate(CITY_LIST):
        print(f"\n--- Train city: {train_city} ---")
        train_graphs = load_city_graphs(train_city, DATA_DIR)
        train_graphs = undersample_graphs(train_graphs, rng)
        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)

        sample = train_graphs[0]
        in_dim = sample.x.size(1)
        edge_dim = sample.edge_attr.size(1)
        num_classes = int(torch.cat([g.y for g in train_graphs]).max().item() + 1)
        model = EgoGCN(in_dim, HIDDEN_DIM, num_classes, edge_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(train_loader, model, optimizer, criterion, DEVICE)

        # Test on all cities
        for j, test_city in enumerate(CITY_LIST):
            test_graphs = load_city_graphs(test_city, DATA_DIR)
            test_graphs = undersample_graphs(test_graphs, rng)
            test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)
            acc, f1, prec, rec = eval_epoch(test_loader, model, DEVICE)
            results[i, j] = acc
            print(f"Train {train_city} -> Test {test_city} | Acc {acc:.4f} | F1 {f1:.4f}")

    print("\n=== Cross-City Accuracy Matrix ===")
    import pandas as pd
    df = pd.DataFrame(results, index=CITY_LIST, columns=CITY_LIST)
    display(df)
    return df

# Run everything
for city in CITY_LIST:
    run_within_city(city)

run_leave_one_out()
cross_city_results = run_cross_city()
