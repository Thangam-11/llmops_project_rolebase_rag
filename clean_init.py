from pathlib import Path

target = Path("src/ingestion_pipeline/__init__.py")

# Read current content
current = target.read_text()
print(f"Current size : {len(current)} bytes")
print(f"Current content:\n{current[:200]}")

# Clear it
target.write_text("")

# Verify
after = target.read_text()
print(f"\nAfter size   : {len(after)} bytes")

if len(after) == 0:
    print("✅ File cleared successfully")
else:
    print("❌ File still has content")