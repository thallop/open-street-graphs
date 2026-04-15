# build_city_graphs.py

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def extract_urban_mask(raster_paths, city_gdf, threshold=0):
    """
    Builds a merged polygon of urban areas from raster files overlapping the city.

    Parameters
    ----------
    raster_paths : list[str]
        Paths to raster files (.tif)
    city_gdf : GeoDataFrame
        Georeferenced city geometry (same CRS as nodes)
    threshold : float
        Keep pixels strictly greater than this value

    Returns
    -------
    GeoDataFrame
        Dissolved urban polygons in the same CRS as city_gdf
    """
    if city_gdf.empty or city_gdf.crs is None:
        raise ValueError("city_gdf must be non-empty and georeferenced.")

    city_crs = city_gdf.crs
    city_geom = city_gdf.union_all().convex_hull
    parts = []

    for path in raster_paths:
        with rasterio.open(path) as src:
            city_geom_src = gpd.GeoSeries([city_geom], crs=city_crs).to_crs(src.crs).iloc[0]
            if not geometry.box(*src.bounds).intersects(city_geom_src):
                continue

            out_image, out_transform = mask(src, [geometry.mapping(city_geom_src)], crop=True)
            band = out_image[0]

            if src.nodata is not None:
                valid = (band != src.nodata)
            else:
                valid = ~np.isnan(band)

            binary_mask = (band > threshold) & valid
            if not binary_mask.any():
                continue

            polygons = [
                geometry.shape(geom)
                for geom, val in shapes(binary_mask.astype(np.uint8),
                                        mask=binary_mask,
                                        transform=out_transform)
                if val == 1
            ]
            if not polygons:
                continue
            gdf_part = gpd.GeoDataFrame(geometry=polygons, crs=src.crs)
            parts.append(gdf_part)

    if not parts:
        raise ValueError("No raster overlaps the city geometry or threshold removed all pixels.")

    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True)).dissolve().reset_index(drop=True)
    return merged.to_crs(city_crs)

    
def compute_lengths_in_meters(nodes_gdf, edges_gdf):
    """
    Compute edge lengths in meters using automatic projected CRS.
    Returns updated edges and the metric CRS used.
    """
    try:
        metric_crs = nodes_gdf.estimate_utm_crs()
    except Exception:
        warnings.warn("UTM estimation failed. Falling back to EPSG:3857.")
        metric_crs = "EPSG:3857"

    nodes_m = nodes_gdf.to_crs(metric_crs)
    edges_m = edges_gdf.to_crs(metric_crs)

    edges_gdf["length_m"] = edges_m.geometry.length.astype(float)

    node_geom = nodes_m.set_index("osmid")["geometry"].to_dict()
    abs_lengths = []
    for _, row in edges_gdf.iterrows():
        u, v = row["u"], row["v"]
        if (u in node_geom) and (v in node_geom):
            abs_lengths.append(node_geom[u].distance(node_geom[v]))
        else:
            abs_lengths.append(np.nan)
    edges_gdf["length_abs_m"] = abs_lengths

    return edges_gdf, metric_crs

    
def compute_offsets_in_meters(nodes_gdf, edges_gdf, metric_crs):
    """
    Compute offset metrics (distance to barycenter of neighbors) in meters.

    Returns updated nodes with:
    - offset_norm_m : distance magnitude in meters
    - offset_cos    : cosine of the offset angle
    - offset_sin    : sine of the offset angle
    """
    nodes_m = nodes_gdf.to_crs(metric_crs)
    edges_m = edges_gdf.to_crs(metric_crs)

    adjacency = {nid: [] for nid in nodes_gdf["osmid"]}
    for u, v in edges_m[["u", "v"]].itertuples(index=False):
        if u in adjacency:
            adjacency[u].append(v)
        if v in adjacency:
            adjacency[v].append(u)

    coords_m = nodes_m.set_index("osmid")["geometry"].apply(lambda p: (p.x, p.y)).to_dict()

    offset_norm, offset_cos, offset_sin = [], [], []
    for nid, geom in nodes_m.set_index("osmid")["geometry"].items():
        neighs = [n for n in adjacency.get(nid, []) if n in coords_m]
        if not neighs:
            offset_norm.append(0.0)
            offset_cos.append(0.0)
            offset_sin.append(0.0)
            continue

        neigh_pts = np.array([coords_m[n] for n in neighs])
        bary_x, bary_y = neigh_pts.mean(axis=0)
        dx, dy = bary_x - geom.x, bary_y - geom.y

        norm = float(np.hypot(dx, dy))
        angle = float(np.arctan2(dy, dx))

        offset_norm.append(norm)
        offset_cos.append(np.cos(angle))
        offset_sin.append(np.sin(angle))

    nodes_gdf["offset_norm_m"] = offset_norm
    nodes_gdf["offset_cos"] = offset_cos
    nodes_gdf["offset_sin"] = offset_sin
    return nodes_gdf

    
def process_city(city_name, outputs_stage0, favelas_path, raster_paths, output_dir):
    safe_name = city_name.lower().replace(",", "").replace(" ", "_")
    nodes_path = os.path.join(outputs_stage0, f"nodes_{safe_name}.gpkg")
    edges_path = os.path.join(outputs_stage0, f"edges_{safe_name}.gpkg")

    nodes = gpd.read_file(nodes_path)
    edges = gpd.read_file(edges_path)
    favelas = gpd.read_file(favelas_path)

    base_crs = nodes.crs
    if favelas.crs != base_crs:
        favelas = favelas.to_crs(base_crs)

    print(f"--- {city_name} ---")
    print("Extracting urban polygons ...")
    urban = extract_urban_mask(raster_paths, nodes, threshold=0)
    if urban.crs != base_crs:
        urban = urban.to_crs(base_crs)

    # Label class = 1 if node is inside a favela
    nodes["_tmp_id"] = np.arange(len(nodes))
    nodes_in_fav = gpd.sjoin(nodes[["geometry", "_tmp_id"]],
                             favelas[["geometry"]],
                             how="left", predicate="within")
    nodes["class"] = (~nodes_in_fav["index_right"].isna()).astype(int)
    nodes.drop(columns=["_tmp_id"], inplace=True)

    # Filter nodes inside urban areas
    nodes["_tmp_id"] = np.arange(len(nodes))
    nodes_in_urban = gpd.sjoin(nodes[["geometry", "_tmp_id"]],
                               urban[["geometry"]],
                               how="left", predicate="within")
    keep = ~nodes_in_urban["index_right"].isna()
    nodes = nodes.loc[keep.values].drop(columns=["_tmp_id"]).copy()
    nodes.reset_index(drop=True, inplace=True)

    valid_osmids = set(nodes["osmid"])
    edges = edges[edges["u"].isin(valid_osmids) & edges["v"].isin(valid_osmids)].copy()
    edges.reset_index(drop=True, inplace=True)

    # Compute spatial metrics
    edges, metric_crs = compute_lengths_in_meters(nodes, edges)
    nodes = compute_offsets_in_meters(nodes, edges, metric_crs)

    # Remove isolated nodes
    connected_nodes = set(edges["u"]) | set(edges["v"])
    before = len(nodes)
    nodes = nodes[nodes["osmid"].isin(connected_nodes)].copy()
    nodes.reset_index(drop=True, inplace=True)
    print(f"Removed {before - len(nodes)} isolated nodes.")

    # Build PyTorch Geometric graph
    embed_cols = [f"A{str(i).zfill(2)}" for i in range(64)]
    geom_cols = ["offset_norm_m", "offset_cos", "offset_sin"]
    edge_feat_cols = ["length_m", "length_abs_m"]
    
    scaler_geom = StandardScaler()
    nodes[geom_cols] = scaler_geom.fit_transform(nodes[geom_cols])
    
    scaler_edges = StandardScaler()
    edges[edge_feat_cols] = scaler_edges.fit_transform(edges[edge_feat_cols])
    
    id_to_idx = {nid: i for i, nid in enumerate(nodes["osmid"])}
    edge_index_list = [
        [id_to_idx[u], id_to_idx[v]]
        for u, v in edges[["u", "v"]].itertuples(index=False)
        if u in id_to_idx and v in id_to_idx
    ]
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    y = torch.tensor(nodes["class"].astype(int).to_numpy(), dtype=torch.long)


    # (1) Google-only graph
    x_google = torch.tensor(nodes[embed_cols].to_numpy(), dtype=torch.float)
    edge_attr_dummy = torch.zeros((edges.shape[0], 1), dtype=torch.float)
    graph_google = Data(x=x_google, edge_index=edge_index, edge_attr=edge_attr_dummy, y=y)

    # (2) Geometry-only graph
    x_geom = torch.tensor(nodes[geom_cols].to_numpy(), dtype=torch.float)
    edge_attr_geom = torch.tensor(edges[edge_feat_cols].to_numpy(), dtype=torch.float)
    graph_geom = Data(x=x_geom, edge_index=edge_index, edge_attr=edge_attr_geom, y=y)

    # (3) Full graph (embeddings + geometry)
    node_feat_cols = embed_cols + geom_cols
    x_full = torch.tensor(nodes[node_feat_cols].to_numpy(), dtype=torch.float)
    edge_attr_full = torch.tensor(edges[edge_feat_cols].to_numpy(), dtype=torch.float)
    graph_full = Data(x=x_full, edge_index=edge_index, edge_attr=edge_attr_full, y=y)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = city_name.lower().replace(",", "").replace(" ", "_")

    # Save only the full version as GeoPackage
    nodes_out = nodes[["osmid", "geometry", "class"] + node_feat_cols].copy()
    edges_out = edges[["u", "v", "key", "geometry"] + edge_feat_cols].copy()
    nodes_out.to_file(os.path.join(output_dir, f"nodes_{safe_name}_full.gpkg"), driver="GPKG")
    edges_out.to_file(os.path.join(output_dir, f"edges_{safe_name}_full.gpkg"), driver="GPKG")

    # Save all three PyG graphs
    torch.save(graph_google, os.path.join(output_dir, f"graph_{safe_name}_google_only.pt"))
    torch.save(graph_geom, os.path.join(output_dir, f"graph_{safe_name}_geometry_only.pt"))
    torch.save(graph_full, os.path.join(output_dir, f"graph_{safe_name}_full.pt"))

    print(f"Saved graphs for {city_name}:")
    print(f"  graph_{safe_name}_google_only.pt")
    print(f"  graph_{safe_name}_geometry_only.pt")
    print(f"  graph_{safe_name}_full.pt")
    print(f"  (GeoPackages stored only for full version)\n")
    '''   
    # Normalize node and edge features
    embed_cols = [f"A{str(i).zfill(2)}" for i in range(64)]
    geom_cols = ["offset_norm_m", "offset_cos", "offset_sin"]
    scaler_geom = StandardScaler()
    nodes[geom_cols] = scaler_geom.fit_transform(nodes[geom_cols])

    edge_feat_cols = ["length_m", "length_abs_m"]
    scaler_edges = StandardScaler()
    edges[edge_feat_cols] = scaler_edges.fit_transform(edges[edge_feat_cols])

    node_feat_cols = embed_cols + geom_cols

    id_to_idx = {nid: i for i, nid in enumerate(nodes["osmid"])}
    edge_index_list = []
    for u, v in edges[["u", "v"]].itertuples(index=False):
        if u in id_to_idx and v in id_to_idx:
            edge_index_list.append([id_to_idx[u], id_to_idx[v]])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    x = torch.tensor(nodes[node_feat_cols].to_numpy(), dtype=torch.float)
    y = torch.tensor(nodes["class"].astype(int).to_numpy(), dtype=torch.long)
    edge_attr = torch.tensor(edges[edge_feat_cols].to_numpy(), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Save outputs (unique filenames)
    nodes_out = nodes[["osmid", "geometry", "class"] + node_feat_cols].copy()
    edges_out = edges[["u", "v", "key", "geometry"] + edge_feat_cols].copy()

    os.makedirs(output_dir, exist_ok=True)

    nodes_path_out = os.path.join(output_dir, f"nodes_{safe_name}_clean.gpkg")
    edges_path_out = os.path.join(output_dir, f"edges_{safe_name}_clean.gpkg")
    graph_path_out = os.path.join(output_dir, f"graph_{safe_name}_clean.pt")

    nodes_out.to_file(nodes_path_out, driver="GPKG")
    edges_out.to_file(edges_path_out, driver="GPKG")
    torch.save(data, graph_path_out)

    print(f"Saved:")
    print(f"  {nodes_path_out}")
    print(f"  {edges_path_out}")
    print(f"  {graph_path_out}\n")
    '''

path_outputs_stage0 = "./data/raw_graphs"  # folder containing enriched nodes_<city>.gpkg and edges_<city>.gpkg
ground_truth = "./data/FCU_IBGE_22.gpkg"  
raster_dir = "./data/GHSL" # built-up surface
raster_paths = [os.path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith(".tif")]
output_dir = "./data/clean_graphs"

city_list = [
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
    "Porto Alegre, Brazil"
    ]

for city in city_list:
    process_city(city, path_outputs_stage0, ground_truth, raster_paths, output_dir)
