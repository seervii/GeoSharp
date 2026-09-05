"""
Download sample Sentinel-2 L2A tiles from the Copernicus Data Space Ecosystem.

For the demo you likely only need 1-3 tiles, hand-picked for good visual
contrast (e.g. one agricultural area, one with clear cloud-free coverage).
It's usually faster to browse and download a couple of tiles manually via
https://browser.dataspace.copernicus.eu/ for the demo, and only script the
download if you need to automate pulling many tiles.

This script is a starting point for the scripted/API path using the
Copernicus Data Space Ecosystem's OData/STAC API.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env file in project root if present

COPERNICUS_AUTH_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/"
    "openid-connect/token"
)
COPERNICUS_SEARCH_URL = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
)


def get_access_token(username: str, password: str) -> str:
    """Authenticate with Copernicus Data Space Ecosystem, return access token."""
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    response = requests.post(COPERNICUS_AUTH_URL, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


def search_tiles(bbox: str, date_start: str, date_end: str, max_cloud: int = 10):
    """
    Search for Sentinel-2 L2A tiles matching a bounding box, date range,
    and max cloud cover percentage.

    bbox format: "min_lon,min_lat,max_lon,max_lat"
    date format: "YYYY-MM-DD"
    """
    filter_query = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({bbox}))') "
        f"and ContentDate/Start gt {date_start}T00:00:00.000Z "
        f"and ContentDate/Start lt {date_end}T00:00:00.000Z "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq "
        f"'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {max_cloud})"
    )
    params = {"$filter": filter_query, "$top": 5}
    response = requests.get(COPERNICUS_SEARCH_URL, params=params)
    response.raise_for_status()
    return response.json().get("value", [])


def download_tile(product_id: str, token: str, out_dir: str = "data/raw"):
    """Download a single product by its Copernicus product ID."""
    os.makedirs(out_dir, exist_ok=True)
    url = (
        f"https://zipper.dataspace.copernicus.eu/odata/v1/"
        f"Products({product_id})/$value"
    )
    headers = {"Authorization": f"Bearer {token}"}
    out_path = os.path.join(out_dir, f"{product_id}.zip")

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Downloaded: {out_path}")
    return out_path


if __name__ == "__main__":
    print(
        "For the demo, manually downloading 1-3 tiles from "
        "https://browser.dataspace.copernicus.eu/ is usually faster than "
        "scripting this. Use this module's functions if you need to "
        "automate pulling multiple tiles.\n"
    )
    print(
        "Example usage (reads credentials from .env):\n"
        "  username = os.environ['COPERNICUS_USERNAME']\n"
        "  password = os.environ['COPERNICUS_PASSWORD']\n"
        "  token = get_access_token(username, password)\n"
        "  results = search_tiles(bbox='...', date_start='2026-01-01', "
        "date_end='2026-02-01')\n"
        "  download_tile(results[0]['Id'], token)"
    )
