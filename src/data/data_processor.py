import pandas as pd  
import glob 
from tqdm import tqdm
import os 

source_dir = "data/raw"
extract_dir = "data/processed"

os.makedirs(extract_dir, exist_ok=True)

columns = ['TA_F', 'PA_F', 'WS_F', 'CO2_F_MDS', 'VPD_F'] 

files = glob.glob(source_dir + "/*.csv")
for file in tqdm(files): 
    site = file.split("_")[1]

    df = pd.read_csv(file)
    df = df[(df['TIMESTAMP_START'] >= 201001040000) & (df['TIMESTAMP_START'] < 201312300000)]
    df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"], format="%Y%m%d%H%M")
    df['week'] = df["TIMESTAMP_START"].dt.to_period("W")
    
    corr = df.groupby("week")[columns].apply(lambda x: x.corr())
    corr.to_csv(extract_dir + f"/{site}_corr.csv") 