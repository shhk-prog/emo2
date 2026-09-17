import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd
import glob, os

cache_dir = 'v1/data/raw/keidolabs___aipsy-affect/default/0.0.0/1e24598f2041e86f788d1974cb3661b67d236fe4'
splits = ['clinical', 'neutral', 'moderate', 'complex_neutral']
dfs = {}

for s in splits:
    arrow_file = os.path.join(cache_dir, f'aipsy-affect-{s}.arrow')
    try:
        with pa.memory_map(arrow_file, 'r') as source:
            reader = ipc.open_stream(source)
            table = reader.read_all()
    except Exception:
        with pa.memory_map(arrow_file, 'r') as source:
            reader = ipc.open_file(source)
            table = reader.read_all()
    df = table.to_pandas()
    df['split'] = s
    dfs[s] = df
    print(f"Split {s}: {len(df)} rows. Columns: {list(df.columns)}")
    print(df[['id', 'emotion', 'intensity', 'matched_control_id']].head(2))

all_df = pd.concat(dfs.values(), ignore_index=True)
print("\nTotal rows:", len(all_df))
print("\nEmotion counts:")
print(all_df.groupby(['split', 'emotion']).size().unstack(fill_value=0))
