"""
GeoSharp - Indian Sentinel-2 L2A Downloader

Downloads exactly 2 Sentinel-2 L2A products for each category:

    1. Agriculture
    2. Urban
    3. Coastal
    4. Forest
    5. Mixed

Maximum total = 10 products.

Project structure:

    prototype/
    ├── .env
    ├── pipeline/
    │   └── download.py
    ├── data/
    │   └── raw/
    └── outputs/

.env:

    COPERNICUS_USERNAME=your_email
    COPERNICUS_PASSWORD=your_password
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================================
# LOAD .ENV
# ============================================================================

# download.py is inside:
#
# prototype/pipeline/download.py
#
# Therefore:
# parent       = pipeline
# parent.parent = prototype

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================================
# COPERNICUS API
# ============================================================================

COPERNICUS_AUTH_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

COPERNICUS_SEARCH_URL = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
)

COPERNICUS_DOWNLOAD_URL = (
    "https://zipper.dataspace.copernicus.eu/"
    "odata/v1/Products({product_id})/$value"
)


# ============================================================================
# SETTINGS
# ============================================================================

MAX_CLOUD = 10

# Search period
DATE_START = "2025-01-01"
DATE_END = "2026-09-05"

# Exactly 2 products per category
PRODUCTS_PER_CATEGORY = 2

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================================
# INDIAN REGIONS
# ============================================================================
#
# bbox format:
#
# (minimum longitude,
#  minimum latitude,
#  maximum longitude,
#  maximum latitude)
#
# ============================================================================

REGIONS = {

    "agriculture": {
        "name": "Punjab agriculture",
        "bbox": (75.3, 30.5, 75.8, 30.9),
    },

    "urban": {
        "name": "Delhi urban",
        "bbox": (77.0, 28.4, 77.5, 28.8),
    },

    "coastal": {
        "name": "Mumbai coastal",
        "bbox": (72.7, 18.8, 73.1, 19.3),
    },

    "forest": {
        "name": "Western Ghats",
        "bbox": (73.5, 15.0, 74.0, 15.5),
    },

    "mixed": {
        "name": "Hyderabad mixed",
        "bbox": (78.2, 17.2, 78.7, 17.7),
    },
}


# ============================================================================
# AUTHENTICATION
# ============================================================================

def get_access_token(username: str, password: str) -> str:
    """
    Authenticate with Copernicus Data Space Ecosystem.
    """

    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    print("Authenticating with Copernicus...")

    response = requests.post(
        COPERNICUS_AUTH_URL,
        data=data,
        timeout=60,
    )

    response.raise_for_status()

    token = response.json()["access_token"]

    print("Authentication successful.\n")

    return token


# ============================================================================
# BBOX -> POLYGON
# ============================================================================

def bbox_to_polygon(bbox):
    """
    Convert bounding box into OData polygon coordinates.
    """

    min_lon, min_lat, max_lon, max_lat = bbox

    return (
        f"{min_lon} {min_lat},"
        f"{max_lon} {min_lat},"
        f"{max_lon} {max_lat},"
        f"{min_lon} {max_lat},"
        f"{min_lon} {min_lat}"
    )


# ============================================================================
# SEARCH SENTINEL-2 L2A
# ============================================================================

def search_tiles(
    bbox,
    date_start=DATE_START,
    date_end=DATE_END,
    max_cloud=MAX_CLOUD,
    max_results=2,
):
    """
    Search for Sentinel-2 L2A products.

    Returns at most max_results products.
    """

    polygon = bbox_to_polygon(bbox)

    filter_query = (
        "Collection/Name eq 'SENTINEL-2' and "

        # Geographic intersection
        f"OData.CSC.Intersects("
        f"area=geography'SRID=4326;"
        f"POLYGON(({polygon}))'"
        f") and "

        # Date range
        f"ContentDate/Start ge "
        f"{date_start}T00:00:00.000Z and "

        f"ContentDate/Start le "
        f"{date_end}T23:59:59.999Z and "

        # Explicitly require Sentinel-2 L2A
        "Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' and "
        "att/OData.CSC.StringAttribute/Value eq 'S2MSI2A'"
        ") and "

        # Cloud cover
        "Attributes/OData.CSC.DoubleAttribute/any("
        "att:att/Name eq 'cloudCover' and "
        f"att/OData.CSC.DoubleAttribute/Value le {max_cloud}"
        ")"
    )

    params = {
        "$filter": filter_query,
        "$orderby": "ContentDate/Start desc",
        "$top": max_results,
    }

    response = requests.get(
        COPERNICUS_SEARCH_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json().get("value", [])


# ============================================================================
# DOWNLOAD PRODUCT
# ============================================================================

def download_tile(
    product,
    token,
    category,
):
    """
    Download one Sentinel-2 product as a ZIP file.
    """

    category_dir = OUTPUT_DIR / category

    category_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    product_id = product["Id"]

    product_name = product.get(
        "Name",
        product_id,
    )

    # Safe filename
    safe_name = "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in product_name
    )

    output_path = category_dir / f"{safe_name}.zip"

    # ------------------------------------------------------------
    # Already downloaded?
    # ------------------------------------------------------------

    if output_path.exists():

        print(
            f"Already exists:\n"
            f"  {output_path}"
        )

        return output_path

    # ------------------------------------------------------------
    # Download URL
    # ------------------------------------------------------------

    url = COPERNICUS_DOWNLOAD_URL.format(
        product_id=product_id
    )

    headers = {
        "Authorization": f"Bearer {token}"
    }

    print("\n" + "-" * 70)

    print(
        f"Downloading:\n"
        f"  {product_name}"
    )

    print(
        f"Category:\n"
        f"  {category}"
    )

    print(
        f"Output:\n"
        f"  {output_path}"
    )

    print("-" * 70)

    # ------------------------------------------------------------
    # Stream download
    # ------------------------------------------------------------

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=120,
    ) as response:

        response.raise_for_status()

        total_size = int(
            response.headers.get(
                "content-length",
                0,
            )
        )

        downloaded = 0

        with open(
            output_path,
            "wb",
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024,
            ):

                if not chunk:
                    continue

                file.write(chunk)

                downloaded += len(chunk)

                # Progress
                if total_size > 0:

                    percent = (
                        downloaded
                        / total_size
                        * 100
                    )

                    print(
                        f"\rProgress: "
                        f"{percent:6.2f}%",
                        end="",
                    )

    print(
        "\nDownload complete."
    )

    return output_path


# ============================================================================
# PRINT PRODUCT INFORMATION
# ============================================================================

def print_product_info(product):
    """
    Print useful information about a Copernicus product.
    """

    name = product.get(
        "Name",
        "Unknown",
    )

    product_id = product.get(
        "Id",
        "Unknown",
    )

    print(
        f"\nProduct: {name}"
    )

    print(
        f"ID: {product_id}"
    )

    # Try to extract cloud cover
    cloud_cover = None

    for attribute in product.get(
        "Attributes",
        [],
    ):

        if attribute.get("Name") == "cloudCover":

            cloud_cover = attribute.get(
                "Value"
            )

            break

    if cloud_cover is not None:

        print(
            f"Cloud cover: {cloud_cover}%"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "\n"
        "==============================================\n"
        "     GeoSharp - Indian Sentinel-2 Downloader\n"
        "==============================================\n"
    )

    print(
        f"Project root:\n"
        f"  {PROJECT_ROOT}"
    )

    print(
        f"\n.env file:\n"
        f"  {ENV_FILE}"
    )

    # ========================================================================
    # CHECK CREDENTIALS
    # ========================================================================

    username = os.getenv(
        "COPERNICUS_USERNAME"
    )

    password = os.getenv(
        "COPERNICUS_PASSWORD"
    )

    if not username or not password:

        raise RuntimeError(
            "\n"
            "Copernicus credentials not found.\n\n"
            f"Expected .env file at:\n"
            f"{ENV_FILE}\n\n"
            "It should contain:\n\n"
            "COPERNICUS_USERNAME=your_username\n"
            "COPERNICUS_PASSWORD=your_password\n"
        )

    print(
        "\nCredentials found."
    )

    # ========================================================================
    # AUTHENTICATE
    # ========================================================================

    token = get_access_token(
        username,
        password,
    )

    # ========================================================================
    # DOWNLOAD 2 PRODUCTS PER CATEGORY
    # ========================================================================

    total_downloaded = 0

    for category, region in REGIONS.items():

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"CATEGORY: {category.upper()}"
        )

        print(
            f"Region: {region['name']}"
        )

        print(
            "=" * 70
        )

        # ------------------------------------------------------------
        # Search
        # ------------------------------------------------------------

        print(
            "\nSearching Sentinel-2 L2A products..."
        )

        try:

            results = search_tiles(
                bbox=region["bbox"],
                max_cloud=MAX_CLOUD,
                max_results=PRODUCTS_PER_CATEGORY,
            )

        except Exception as error:

            print(
                f"\nSearch failed:\n"
                f"{error}"
            )

            continue

        # ------------------------------------------------------------
        # No results
        # ------------------------------------------------------------

        if not results:

            print(
                "\nNo suitable products found."
            )

            continue

        # ------------------------------------------------------------
        # Display results
        # ------------------------------------------------------------

        print(
            f"\nFound {len(results)} product(s)."
        )

        for index, product in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n[{index}]"
            )

            print_product_info(
                product
            )

        # ------------------------------------------------------------
        # Download
        # ------------------------------------------------------------

        for product in results[
            :PRODUCTS_PER_CATEGORY
        ]:

            try:

                download_tile(
                    product=product,
                    token=token,
                    category=category,
                )

                total_downloaded += 1

            except Exception as error:

                print(
                    f"\nDownload failed:\n"
                    f"{error}"
                )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print(
        "\n\n"
        "==============================================\n"
        "              DOWNLOAD COMPLETE\n"
        "=============================================="
    )

    print(
        f"\nProducts downloaded: "
        f"{total_downloaded}"
    )

    print(
        f"\nMaximum expected: "
        f"{len(REGIONS) * PRODUCTS_PER_CATEGORY}"
    )

    print(
        "\nOutput directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\nFolder structure:"
    )

    for category in REGIONS:

        print(
            f"  data\\raw\\{category}\\"
        )

    print(
        "\n==============================================\n"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()