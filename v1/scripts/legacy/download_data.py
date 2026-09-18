#!/usr/bin/env python3
"""Downloads the EmoBank dataset from its public repository."""
import os
import requests
import urllib.request
import zipfile

def download_emobank(output_dir: str = "v1/data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv"
    dest_path = os.path.join(output_dir, "emobank.csv")
    
    print(f"Downloading EmoBank from {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved to {dest_path}")

if __name__ == "__main__":
    download_emobank()
