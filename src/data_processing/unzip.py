import os
import zipfile
import glob

zip_dir = 'data/zip'
extract_dir = 'data/raw'

zip_files = glob.glob(zip_dir + '/*.zip')

os.makedirs(extract_dir, exist_ok=True)

for zip_file in zip_files:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if "_HH_" in file and file.endswith(".csv"):
                zip_ref.extract(file, extract_dir)
                print(f"Extracted {file} from {zip_file} to {extract_dir}")