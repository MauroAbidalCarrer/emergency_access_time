from pathlib import Path
import base64
import colorsys
import html
from io import BytesIO
import math

import networkx as nx
import numpy as np
import osmnx as ox
import folium
import streamlit as st
from PIL import Image, ImageDraw
from pyrosm import OSM
from streamlit_folium import st_folium


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "osm_cache"
PBF_PATH = DATA_DIR / "osm" / "ile-de-france.osm.pbf"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

ox.settings.use_cache = True
ox.settings.cache_folder = str(CACHE_DIR)
ox.settings.requests_timeout = 180

LEGEND_MAX_MINUTES = 60.0
RASTER_WIDTH = 3072
RASTER_HEIGHT = 3072


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def rainbow_colour(minutes):
    ratio = min(max(float(minutes) / LEGEND_MAX_MINUTES, 0.0), 1.0)
    # Red -> yellow -> green -> cyan -> blue -> violet.
    hue = 0.75 * (1.0 - ratio)
    red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (
        round(red * 255),
        round(green * 255),
        round(blue * 255),
        205,
    )


def mercator_y(latitude):
    latitude = max(min(float(latitude), 85.05112878), -85.05112878)
    lat_rad = math.radians(latitude)
    return math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0))


def render_edges_to_png(edges):
    """Rasterize coloured road geometries so the browser renders one image."""
    west, south, east, north = edges.total_bounds
    if not all(np.isfinite([west, south, east, north])) or west == east or south == north:
        raise ValueError("Could not determine valid map bounds from road geometries.")

    north_y = mercator_y(north)
    south_y = mercator_y(south)
    if north_y == south_y:
        raise ValueError("Invalid latitude range for raster rendering.")

    image = Image.new("RGBA", (RASTER_WIDTH, RASTER_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")

    def to_pixel(lon, lat):
        x = (float(lon) - west) / (east - west) * (RASTER_WIDTH - 1)
        y = (north_y - mercator_y(lat)) / (north_y - south_y) * (RASTER_HEIGHT - 1)
        return (int(round(x)), int(round(y)))

    for geometry, minutes in zip(edges.geometry, edges.time_min):
        if geometry is None or geometry.is_empty:
            continue
        colour = rainbow_colour(minutes)

        if geometry.geom_type == "LineString":
            parts = [geometry]
        elif geometry.geom_type == "MultiLineString":
            parts = geometry.geoms
        else:
            continue

        for part in parts:
            coords = list(part.coords)
            if len(coords) >= 2:
                pixels = [to_pixel(lon, lat) for lon, lat, *_ in coords]
                draw.line(pixels, fill=colour, width=2)

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    png_bytes = buffer.getvalue()
    image_data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

    # ImageOverlay maps the image corners in Web Mercator. These bounds
    # correspond to the same geographic extent used by to_pixel().
    bounds = [[south, west], [north, east]]
    return image_data_uri, bounds, len(png_bytes)


# --------------------------------------------------
# Hospital and road-network loading
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading the road network and calculating hospital accessibility...")
def load_hospital_accessibility():
    print("Loading OSM data and building road graph...", flush=True)

    if not PBF_PATH.exists():
        raise FileNotFoundError(f"OSM PBF file not found: {PBF_PATH}")

    osm = OSM(str(PBF_PATH))

    hospitals = osm.get_pois(custom_filter={"amenity": ["hospital"]})
    if hospitals is None or hospitals.empty:
        raise ValueError("No hospitals tagged amenity=hospital were found in the PBF file.")

    hospitals = hospitals[
        (hospitals["amenity"] == "hospital")
        & hospitals.geometry.notna()
        & ~hospitals.geometry.is_empty
    ].copy()

    if hospitals.empty:
        raise ValueError("No usable hospital geometries were found.")

    if hospitals.crs is not None and hospitals.crs.to_epsg() != 4326:
        hospitals = hospitals.to_crs(epsg=4326)

    hospitals["geometry"] = hospitals.geometry.representative_point()
    hospitals = hospitals[hospitals.geometry.geom_type == "Point"].copy()

    G = osm.get_graph(
        network_type="driving",
        graph_type="networkx",
        retain_all=True,
    )
    G = ox.add_edge_speeds(G, hwy_speeds=100, fallback=30)
    G = ox.add_edge_travel_times(G)

    print(
        f"Loaded {len(G.nodes):,} nodes and {len(G.edges):,} edges",
        flush=True,
    )

    hospital_nodes = ox.distance.nearest_nodes(
        G,
        X=hospitals.geometry.x.to_numpy(),
        Y=hospitals.geometry.y.to_numpy(),
    )
    hospital_nodes = list(set(np.asarray(hospital_nodes).tolist()))
    if not hospital_nodes:
        raise ValueError("Could not connect any hospital to the road network.")

    # Reverse the directed graph so multi-source Dijkstra computes
    # original-direction travel times FROM every node TO a hospital.
    reverse_graph = G.reverse(copy=False)
    time_to_hospital_seconds = nx.multi_source_dijkstra_path_length(
        reverse_graph,
        sources=hospital_nodes,
        weight="travel_time",
    )

    print("Preparing raster accessibility layer...", flush=True)
    edges = ox.graph_to_gdfs(
        G,
        nodes=False,
        edges=True,
        fill_edge_geometry=True,
    )

    u_values = edges.index.get_level_values(0)
    v_values = edges.index.get_level_values(1)

    u_times = np.fromiter(
        (time_to_hospital_seconds.get(node, np.nan) for node in u_values),
        dtype=float,
        count=len(edges),
    )
    v_times = np.fromiter(
        (time_to_hospital_seconds.get(node, np.nan) for node in v_values),
        dtype=float,
        count=len(edges),
    )

    valid_u = np.isfinite(u_times)
    valid_v = np.isfinite(v_times)
    valid = valid_u | valid_v

    edge_times = np.full(len(edges), np.nan, dtype=float)
    both = valid_u & valid_v
    edge_times[both] = (u_times[both] + v_times[both]) / 2.0
    edge_times[valid_u & ~valid_v] = u_times[valid_u & ~valid_v]
    edge_times[~valid_u & valid_v] = v_times[~valid_u & valid_v]

    edges = edges.loc[valid, ["geometry"]].copy()
    edges["time_min"] = edge_times[valid] / 60.0
    edges["geometry"] = edges.geometry.simplify(
        0.000015,
        preserve_topology=False,
    )

    image_data_uri, image_bounds, png_size = render_edges_to_png(edges)

    reachable_nodes = len(time_to_hospital_seconds)
    max_time_minutes = max(time_to_hospital_seconds.values()) / 60.0

    print(
        f"Found {len(hospitals):,} hospitals; "
        f"{reachable_nodes:,} road nodes can reach a hospital.",
        flush=True,
    )
    print(f"Raster overlay size: {png_size / 1024 / 1024:.1f} MB", flush=True)

    return (
        hospitals,
        image_data_uri,
        image_bounds,
        len(G.nodes),
        len(G.edges),
        reachable_nodes,
        max_time_minutes,
        png_size,
    )


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------

st.set_page_config(page_title="Emergency Accessibility", layout="wide")
st.title("Emergency Accessibility - Île-de-France")
st.write(
    "Road segments are coloured by estimated driving time to the nearest hospital. "
    "The road layer is rendered as a raster image to keep the browser responsive."
)

(
    hospitals,
    accessibility_image,
    image_bounds,
    node_count,
    edge_count,
    reachable_nodes,
    max_time_minutes,
    png_size,
) = load_hospital_accessibility()


# --------------------------------------------------
# Build map
# --------------------------------------------------

m = folium.Map(
    location=[48.85, 2.35],
    zoom_start=9,
    tiles="OpenStreetMap",
    control_scale=True,
)

folium.raster_layers.ImageOverlay(
    image=accessibility_image,
    bounds=image_bounds,
    opacity=1.0,
    name="Driving time to nearest hospital",
    interactive=False,
    cross_origin=False,
    zindex=2,
).add_to(m)

hospital_layer = folium.FeatureGroup(name="Hospitals", show=True)
for _, hospital in hospitals.iterrows():
    point = hospital.geometry
    hospital_name = hospital.get("name")
    if not isinstance(hospital_name, str) or not hospital_name.strip():
        hospital_name = "Hospital (name not available in OSM)"

    folium.CircleMarker(
        location=[point.y, point.x],
        radius=3,
        color="blue",
        weight=1,
        fill=True,
        fill_color="blue",
        fill_opacity=0.9,
        tooltip=html.escape(hospital_name),
        popup=html.escape(hospital_name),
    ).add_to(hospital_layer)

hospital_layer.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)


# Rainbow legend
legend_html = f"""
<div style="
    position: fixed;
    bottom: 28px;
    left: 28px;
    z-index: 9999;
    background: white;
    padding: 10px 12px;
    border: 1px solid #777;
    border-radius: 4px;
    width: 270px;
    font-size: 12px;
">
    <div style="font-weight: bold; margin-bottom: 6px;">
        Estimated driving time to nearest hospital
    </div>
    <div style="
        height: 13px;
        width: 100%;
        background: linear-gradient(
            to right,
            #ff0000,
            #ffff00,
            #00ff00,
            #00ffff,
            #0000ff,
            #8000ff
        );
        border: 1px solid #555;
    "></div>
    <div style="display: flex; justify-content: space-between; margin-top: 4px;">
        <span>0 min</span>
        <span>{LEGEND_MAX_MINUTES:.0f}+ min</span>
    </div>
    <div style="margin-top: 5px; color: #555;">
        Red = shorter time; violet = longer time.
    </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(
    m,
    height=700,
    use_container_width=True,
    returned_objects=[],
    key="hospital_isochrone_map",
)


# --------------------------------------------------
# Network information
# --------------------------------------------------

col1, col2, col3 = st.columns(3)
col1.metric("Hospitals in OSM", f"{len(hospitals):,}")
col2.metric("Road nodes connected to a hospital", f"{reachable_nodes:,}")
col3.metric("Slowest reachable node", f"{max_time_minutes:.1f} min")

with st.expander("Method and network information"):
    st.write(f"Road graph nodes: {node_count:,}")
    st.write(f"Road graph edges: {edge_count:,}")
    st.write(f"Raster overlay size: {png_size / 1024 / 1024:.1f} MB")
    st.write(f"OSM source: {PBF_PATH}")
    st.write(
        "Travel times are estimated from OSM road classes and assigned speeds. "
        "They do not include live traffic, ambulance-specific speed assumptions, "
        "dispatch delays, or hospital capacity."
    )
    st.write(
        "The colours describe the road network. They are not filled "
        "administrative or geographic polygons."
    )
    st.write(
        "Hospitals are OSM features tagged amenity=hospital and snapped "
        "to the nearest drivable road-network node."
    )
    st.write(
        "Road hover tooltips are omitted in this raster version. Hospital "
        "markers remain interactive."
    )
