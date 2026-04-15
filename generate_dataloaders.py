# dataloaders.py

"""
Local graph extraction and DataLoader preparation for multiple cities
and multiple graph versions (google_only, geometry_only, full).
Supports num_hops = 0 (node-only baseline), 1, 2, 3.
"""

import os
import sys
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
from joblib import Parallel, delayed


# ================================================================
# 0) PARAMETERS
# ================================================================
CITIES = [
    "São Paulo, Brazil",
    "Rio de Janeiro, Brazil",
    "Distrito Federal, Brazil",
    "Salvador, Brazil",
    "Fortaleza, Brazil",
    "Belo Horizonte, Brazil",
    "Manaus, Brazil",
    "Curitiba, Brazil",
    "Recife, Brazil",
    "Goiânia, Brazil",
    "Belém, Brazil",
    "Porto Alegre, Brazil"
]

GRAPH_VERSIONS = ["google_only", "geometry_only", "full"]
NUM_HOPS_LIST = [0, 1, 2, 3]

GRAPH_DIR = "/home/hallopeaut/work_ird/open-street-graphs/graph_extraction/geopackages"
OUTPUT_DIR = "/home/hallopeaut/work_ird/open-street-graphs/graph_extraction/dataloaders"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_JOBS = min(len(CITIES), 12)


# ================================================================
# 1) LOCAL GRAPH EXTRACTION
# ================================================================
def extract_local_graph(data: Data, center_node: int, num_hops: int = 2) -> Data:
    """
    Extract a local k-hop subgraph around a given node.
    If num_hops == 0, returns a single-node graph (no edges).
    """
    if num_hops == 0:
        x = data.x[center_node].unsqueeze(0)
        y = data.y[center_node].unsqueeze(0)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        if hasattr(data, "edge_attr") and data.edge_attr.numel() > 0:
            edge_attr = torch.empty((0, data.edge_attr.size(1)), dtype=torch.float)
        else:
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        g.center_idx = 0
        return g

    # Standard ego-graph extraction
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        center_node, num_hops, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
    )
    _, sub_edge_attr = subgraph(subset, data.edge_index, data.edge_attr, relabel_nodes=True)

    mapping = int(mapping)
    if mapping != 0:
        reorder = torch.tensor(
            [mapping] + [i for i in range(len(subset)) if i != mapping],
            dtype=torch.long
        )
        inv = {old: new for new, old in enumerate(reorder.tolist())}
        new_edge_index = sub_edge_index.clone()
        for i in range(new_edge_index.size(1)):
            new_edge_index[0, i] = inv[int(new_edge_index[0, i])]
            new_edge_index[1, i] = inv[int(new_edge_index[1, i])]
        sub_x = data.x[subset][reorder]
        center_idx = 0
    else:
        new_edge_index = sub_edge_index
        sub_x = data.x[subset]
        center_idx = 0

    mini = Data(x=sub_x, edge_index=new_edge_index, edge_attr=sub_edge_attr)
    mini.y = data.y[center_node].unsqueeze(0)
    mini.center_idx = center_idx
    return mini


# ================================================================
# 2) CITY PROCESSING FUNCTION
# ================================================================
def process_city(city_name: str, graph_version: str, num_hops: int):
    """
    Load a city graph (.pt) for a given version and extract local graphs.
    Save them as dataloaders with explicit version + hops.
    """
    safe_name = city_name.lower().replace(",", "").replace(" ", "_")
    graph_path = os.path.join(GRAPH_DIR, f"graph_{safe_name}_{graph_version}.pt")
    output_path = os.path.join(
        OUTPUT_DIR, f"dataloader_{safe_name}_{graph_version}_h{num_hops}.pt"
    )

    if not os.path.exists(graph_path):
        print(f"[{city_name}] Missing graph: {graph_path}")
        return

    if os.path.exists(output_path):
        print(f"[{city_name}] {graph_version} h={num_hops} already processed.")
        return

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    print(
        f"[{city_name}] Loaded {graph_version} graph "
        f"({data.num_nodes} nodes, {data.num_edges} edges) | num_hops={num_hops}"
    )

    local_graphs = [extract_local_graph(data, i, num_hops) for i in range(data.num_nodes)]
    torch.save(local_graphs, output_path)

    print(f"[{city_name}] Saved {len(local_graphs)} local graphs -> {output_path}")


# ================================================================
# 3) MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    import multiprocessing
    torch.set_num_threads(1)  # évite conflits OpenMP sur le cluster

    # Exemple d’appel :
    # python3 pipeline3_local_dataloaders.py "Rio de Janeiro, Brazil" "google_only" 2

    if len(sys.argv) == 4:
        city_arg = sys.argv[1]
        version_arg = sys.argv[2]
        num_hops_arg = int(sys.argv[3])
        print(f"Running local graph extraction for: {city_arg}, version={version_arg}, hops={num_hops_arg}")
        process_city(city_arg, version_arg, num_hops_arg)

    else:
        print("Usage:")
        print("  python3 pipeline3_local_dataloaders.py '<city_name>' <version> <num_hops>")
        print("Example:")
        print("  python3 pipeline3_local_dataloaders.py 'Rio de Janeiro, Brazil' full 2")
        sys.exit(1)

