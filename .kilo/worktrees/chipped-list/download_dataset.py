from pathlib import Path
from ucimlrepo import fetch_ucirepo

OUTPUT = Path("StudentPerformance.csv")

if OUTPUT.exists():
    print(f"{OUTPUT} already exists.")
else:
    print("Downloading Student Performance from UCI...")
    dataset = fetch_ucirepo(id=320)
    df = dataset.data.original.copy()
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT.resolve()}")
