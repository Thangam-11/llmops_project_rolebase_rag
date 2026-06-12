# verify.py
from pathlib import Path

f    = Path("src/ingestion_pipeline/__init__.py")
size = f.stat().st_size
text = f.read_text().strip()

print(f"Size    : {size} bytes")
print(f"Content : '{text}'")

if size == 0:
    print("✅ Empty — ready to run")
else:
    print("❌ Still has content — clear it again")