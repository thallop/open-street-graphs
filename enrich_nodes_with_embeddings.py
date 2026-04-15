# enrich_nodes_with_embeddings.py

import ee
import osmnx as ox
import geopandas as gpd
import pandas as pd
import geemap
from tqdm import tqdm
import os


def initialize_earth_engine():
    """
    Initializes Google Earth Engine.
    Authenticates if necessary.
    """
    try:
        ee.Initialize()
        print("Earth Engine initialized")
    except Exception:
        ee.Authenticate()
        ee.Initialize()
        print("Earth Engine authenticated and initialized")


def load_osm_graph(place_name):
    """
    Loads an OSM graph for a given place.

    Parameters
    ----------
    place_name : str
        Name of the city or area to load.

    Returns
    -------
    nodes_gdf : GeoDataFrame
        Graph nodes with geometry.
    edges_gdf : GeoDataFrame
        Graph edges.
    """
    graph = ox.graph_from_place(
        place_name,
        network_type="all",
        retain_all=True,
        simplify=True
    )
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
    if "osmid" in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.reset_index().rename(columns={"osmid": "osmid"})
    else:
        nodes_gdf = nodes_gdf.reset_index().rename(columns={"index": "osmid"})

    print(f"OSMnx >> {len(nodes_gdf)} nodes, {len(edges_gdf)} edges")
    return nodes_gdf, edges_gdf


def prepare_embedding_image(collection="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL", year=2023):
    """
    Prepares the Earth Engine embedding image for a given year.

    Parameters
    ----------
    collection : str
        The Earth Engine image collection ID.
    year : int
        Year of the embedding.

    Returns
    -------
    ee.Image
        The mosaic image containing embedding bands.
    """
    bands = [f"A{str(i).zfill(2)}" for i in range(64)]

    emb_image = (
        ee.ImageCollection(collection)
        .filterDate(f"{year}-01-01", f"{year+1}-01-01")
        .mosaic()
        .select(bands)
    )

    if emb_image is None:
        raise SystemExit("No embedding image found. Check year or region.")
    print("Image ready")
    return emb_image


def nodes_to_fc(gdf_subset):
    """
    Converts node geometries to a FeatureCollection.

    Parameters
    ----------
    gdf_subset : GeoDataFrame
        Subset of nodes.

    Returns
    -------
    ee.FeatureCollection
        Collection of points with 'osmid' property.
    """
    feats = []
    for _, row in gdf_subset.iterrows():
        osmid = int(row["osmid"])
        lon, lat = float(row.geometry.x), float(row.geometry.y)
        feats.append(ee.Feature(ee.Geometry.Point([lon, lat]), {"osmid": osmid}))
    return ee.FeatureCollection(feats)


def sample_nodes(nodes_gdf, emb_image, batch_size=2000, scale=10):
    """
    Samples embeddings for nodes in batches using geemap.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        Graph nodes.
    emb_image : ee.Image
        Embedding image from Earth Engine.
    batch_size : int
        Number of nodes per batch.
    scale : int
        Pixel resolution in meters.

    Returns
    -------
    DataFrame
        DataFrame containing node embeddings.
    """
    all_dfs = []
    n = len(nodes_gdf)
    print(f"Sampling {n} nodes (batch size {batch_size})")

    for start in tqdm(range(0, n, batch_size)):
        subset = nodes_gdf.iloc[start:start + batch_size]
        fc = nodes_to_fc(subset)
        sampled = emb_image.sampleRegions(fc, scale=scale, geometries=False)
        df = geemap.ee_to_df(sampled)
        if not df.empty:
            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def enrich_city_nodes(place_name, output_dir="./data/raw_graphs", batch_size=2000, year=2023):
    """
    Processes one city: downloads OSM data, samples embeddings, and saves outputs.

    Parameters
    ----------
    place_name : str
        Name of the city or region.
    output_dir : str
        Directory to store results.
    batch_size : int
        Batch size for sampling.
    year : int
        Year for the embedding.

    Returns
    -------
    tuple
        Enriched nodes GeoDataFrame and edges GeoDataFrame.
    """
    nodes_gdf, edges_gdf = load_osm_graph(place_name)
    emb_image = prepare_embedding_image(year=year)
    nodes_embeddings = sample_nodes(nodes_gdf, emb_image, batch_size=batch_size)

    nodes_enriched = nodes_gdf.merge(nodes_embeddings, on="osmid", how="left")
    print(f"Nodes enriched: {nodes_enriched.shape}")

    safe_name = place_name.lower().replace(",", "").replace(" ", "_")
    os.makedirs(output_dir, exist_ok=True)
    nodes_path = f"{output_dir}/nodes_{safe_name}.gpkg"
    edges_path = f"{output_dir}/edges_{safe_name}.gpkg"

    nodes_enriched.to_file(nodes_path, driver="GPKG")
    edges_gdf.to_file(edges_path, driver="GPKG")
    print(f"Saved: {nodes_path} and {edges_path}")

    return nodes_enriched, edges_gdf


initialize_earth_engine()

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
    enrich_city_nodes(city)
