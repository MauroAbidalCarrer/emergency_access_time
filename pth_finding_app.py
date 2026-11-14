from pathlib import Path

from pyrosm import OSM
import osmnx as ox
import folium
import streamlit as st

from streamlit_folium import st_folium


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GRAPH_DIR = DATA_DIR / "graphs"
CACHE_DIR = DATA_DIR / "osm_cache"
PBF_PATH = DATA_DIR / "osm" / "ile-de-france.osm.pbf"

GRAPH_PATH = GRAPH_DIR / "ile_de_france.graphml"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ox.settings.use_cache = True
ox.settings.cache_folder = str(CACHE_DIR)
ox.settings.requests_timeout = 180


# --------------------------------------------------
# Download / load road network
# --------------------------------------------------


import pickle
from pathlib import Path

GRAPH_PATH = GRAPH_DIR / "ile_de_france.pkl"


@st.cache_resource
def load_road_network():

    print("Building graph from PBF...")

    osm = OSM(str(PBF_PATH))

    G = osm.get_graph(
        network_type="driving",
        graph_type="networkx",
        retain_all=True,
    )

    G = ox.add_edge_speeds(
        G,
        hwy_speeds=100,
        fallback=30,
    )

    G = ox.add_edge_travel_times(G)
    print(
        f"Loaded {len(G.nodes):,} nodes "
        f"and {len(G.edges):,} edges"
    )

    return G

# --------------------------------------------------
# Routing
# --------------------------------------------------

def calculate_route(G, point_a, point_b):

    lat_a, lon_a = point_a
    lat_b, lon_b = point_b

    # Find nearest drivable graph nodes.
    origin = ox.distance.nearest_nodes(
        G,
        X=lon_a,
        Y=lat_a,
    )

    destination = ox.distance.nearest_nodes(
        G,
        X=lon_b,
        Y=lat_b,
    )

    # Shortest path based on estimated travel time.
    route = ox.routing.shortest_path(
        G,
        origin,
        destination,
        weight="travel_time",
    )

    if route is None:
        raise ValueError("No route found between these points.")

    # Retrieve actual road geometries.
    route_gdf = ox.routing.route_to_gdf(
        G,
        route,
        weight="travel_time",
    )

    distance_km = route_gdf["length"].sum() / 1000

    travel_time_min = (
        route_gdf["travel_time"].sum() / 60
    )

    return route_gdf, distance_km, travel_time_min


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------

st.set_page_config(
    page_title="Emergency Accessibility",
    layout="wide",
)

st.title("Emergency Accessibility - Île-de-France")

st.write(
    "Select two points on the map to calculate "
    "the fastest road route."
)

G = load_road_network()

# Initialize application state.
if "point_a" not in st.session_state:
    st.session_state.point_a = None

if "point_b" not in st.session_state:
    st.session_state.point_b = None

if "last_map_click" not in st.session_state:
    st.session_state.last_map_click = None


if st.button("Reset selected points"):
    st.session_state.point_a = None
    st.session_state.point_b = None
    st.session_state.last_map_click = None
    st.rerun()


# --------------------------------------------------
# Build map
# --------------------------------------------------

m = folium.Map(
    location=[48.85, 2.35],
    zoom_start=9,
    tiles="OpenStreetMap",
    control_scale=True,
)

if st.session_state.point_a is not None:

    folium.Marker(
        st.session_state.point_a,
        tooltip="Point A",
        popup="Starting point",
        icon=folium.Icon(color="green"),
    ).add_to(m)

if st.session_state.point_b is not None:

    folium.Marker(
        st.session_state.point_b,
        tooltip="Point B",
        popup="Destination",
        icon=folium.Icon(color="red"),
    ).add_to(m)


# --------------------------------------------------
# Calculate and display route
# --------------------------------------------------

if (
    st.session_state.point_a is not None
    and st.session_state.point_b is not None
):

    route_gdf, distance_km, travel_time_min = (
        calculate_route(
            G,
            st.session_state.point_a,
            st.session_state.point_b,
        )
    )

    folium.GeoJson(
        route_gdf.to_json(),
        name="Fastest route",
        style_function=lambda feature: {
            "color": "blue",
            "weight": 5,
            "opacity": 0.8,
        },
    ).add_to(m)

    col1, col2 = st.columns(2)

    col1.metric(
        "Travel time",
        f"{travel_time_min:.1f} min",
    )

    col2.metric(
        "Distance",
        f"{distance_km:.2f} km",
    )


# --------------------------------------------------
# Interactive map click handling
# --------------------------------------------------

st.info(
    "Click once to select point A, "
    "then click again to select point B."
)

map_data = st_folium(
    m,
    height=650,
    use_container_width=True,
    returned_objects=["last_clicked"],
    key="route_map",
)

clicked = map_data.get("last_clicked")

if clicked is not None:

    new_point = (
        round(clicked["lat"], 6),
        round(clicked["lng"], 6),
    )

    if new_point != st.session_state.last_map_click:

        if st.session_state.point_a is None:
            st.session_state.point_a = new_point

        elif st.session_state.point_b is None:
            st.session_state.point_b = new_point

        else:
            # Subsequent clicks replace destination B.
            st.session_state.point_b = new_point

        st.session_state.last_map_click = new_point

        st.rerun()


# --------------------------------------------------
# Network information
# --------------------------------------------------

with st.expander("Road network information"):

    st.write(f"Number of nodes: {len(G.nodes):,}")
    st.write(f"Number of edges: {len(G.edges):,}")
    st.write(f"Graph file: {GRAPH_PATH}")

