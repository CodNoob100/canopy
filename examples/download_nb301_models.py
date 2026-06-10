import urllib.request
import zipfile
import os

url = "https://ndownloader.figshare.com/files/24992018"
zip_path = os.path.expanduser("~/nb301_models_v1.0.zip")
extract_path = os.path.expanduser("~/nb301_models")

print(f"Downloading {url} to {zip_path}... (This is ~1.8GB, please wait)")
urllib.request.urlretrieve(url, zip_path)

print(f"Extracting to {extract_path}...")
os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Only extract the xgboost models to save time and space
    for member in zip_ref.namelist():
        if "xgb" in member or "data_config.json" in member or "model_config.json" in member:
            zip_ref.extract(member, extract_path)

print("Done. Removing zip file...")
os.remove(zip_path)
print("Models successfully installed!")
