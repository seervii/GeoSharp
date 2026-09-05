from pathlib import Path
import sys

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio missing. Run: pip install rasterio")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "opensr_test" / "100" / "naip"

print("=" * 70)
print("GeoSharp - OpenSR-Test NAIP sanity check")
print("=" * 70)
print("Dataset:", DATASET)

if not DATASET.exists():
    print("ERROR: Dataset folder not found.")
    sys.exit(1)

for name in ["hr", "hr_harmonized", "L1C", "L2A"]:
    p = DATASET / name
    print(f"{name:16}:", "FOUND" if p.exists() else "MISSING")

tiffs = sorted(
    p for p in DATASET.rglob("*")
    if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
)

print("\nTIFF files found:", len(tiffs))

if not tiffs:
    print("ERROR: No TIFF files found.")
    sys.exit(1)

# Inspect a few files without touching the 2.7 GB naip.pkl.
print("\nSample TIFFs:")
print("-" * 70)

seen_dirs = set()
samples = []
for p in tiffs:
    if str(p.parent) not in seen_dirs:
        samples.append(p)
        seen_dirs.add(str(p.parent))
    if len(samples) >= 12:
        break

for p in samples:
    try:
        with rasterio.open(p) as src:
            print(p.relative_to(ROOT))
            print("  shape (C,H,W):", (src.count, src.height, src.width))
            print("  resolution   :", src.res)
            print("  dtype        :", src.dtypes)
            print("  bands        :", src.descriptions)
    except Exception as e:
        print(p.relative_to(ROOT), "ERROR:", e)

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)
print("OpenSR-Test NAIP is TEST/benchmark data, not training data.")
print("Do NOT fine-tune on these images.")
print("Next: run GeoSharp on the unseen NAIP pairs and evaluate 10m -> 2.5m.")
print("Then fine-tune separately using KUDALIAR/training data.")
