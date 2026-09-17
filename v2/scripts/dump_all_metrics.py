import json
import glob
import pandas as pd

def main():
    print("=== Dataset Examples ===")
    df = pd.read_csv("v2/data/processed/aipsy_annotated/test_strict.csv")
    pair = df[df['pair_id'] == 'grief-d2-v1']
    if not pair.empty:
        for _, row in pair.iterrows():
            print(f"Condition: {row['condition']}, Intensity: {row['intensity']}")
            print(f"Text: {row['text']}")
            print("-" * 40)
    else:
        for _, row in df.head(3).iterrows():
            print(f"Condition: {row['condition']}, Intensity: {row['intensity']}")
            print(f"Text: {row['text']}")
            print("-" * 40)

    print("\n=== JSON Metrics ===")
    files = glob.glob("v2/results/derived/**/*.json", recursive=True)
    for f in files:
        print(f"\n--- {f} ---")
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error reading {f}: {e}")

if __name__ == "__main__":
    main()
