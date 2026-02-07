import os
import requests
import polars as pl

# --- CONFIGURATION ---
DATA_DIR = "./data"
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

def download_file(url, local_path):
    """Downloads a file if it doesn't already exist locally."""
    if os.path.exists(local_path):
        return True 
    
    print(f"⬇️ [Downloading] {url} ...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024*1024): 
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"❌ [Error] Failed to download {url}: {e}")
        return False

def ingest_and_unify():
    os.makedirs(DATA_DIR, exist_ok=True)
    yellow_files = []
    green_files = []
    
    # 1. DOWNLOAD LOOP (Jan - Dec 2025)
    print("--- Phase 1: Automated Ingestion ---")
    for month in range(1, 13):
        month_str = f"{month:02d}"
        
        # Yellow Taxis
        y_name = f"yellow_tripdata_2025-{month_str}.parquet"
        if download_file(f"{BASE_URL}/{y_name}", f"{DATA_DIR}/{y_name}"):
            yellow_files.append(f"{DATA_DIR}/{y_name}")
            
        # Green Taxis
        g_name = f"green_tripdata_2025-{month_str}.parquet"
        if download_file(f"{BASE_URL}/{g_name}", f"{DATA_DIR}/{g_name}"):
            green_files.append(f"{DATA_DIR}/{g_name}")
            
        # Check for missing December (Constraint)
        if month == 12 and not os.path.exists(f"{DATA_DIR}/{y_name}"):
             print(f"⚠️ Dec 2025 missing. Imputation logic will be applied in pipeline.")

    # 2. SEPARATE SCANS & UNIFICATION
    # We scan yellow and green separately to avoid schema errors, then merge.
    
    # Process Yellow
    q_yellow = pl.scan_parquet(yellow_files).select([
        pl.col("tpep_pickup_datetime").alias("pickup_time"),
        pl.col("tpep_dropoff_datetime").alias("dropoff_time"),
        pl.col("PULocationID").alias("pickup_loc"),
        pl.col("DOLocationID").alias("dropoff_loc"),
        pl.col("trip_distance"),
        pl.col("fare_amount").alias("fare"),
        pl.col("total_amount"),
        pl.col("congestion_surcharge").fill_null(0.0),
        pl.col("VendorID")
    ])

    # Process Green
    q_green = pl.scan_parquet(green_files).select([
        pl.col("lpep_pickup_datetime").alias("pickup_time"),
        pl.col("lpep_dropoff_datetime").alias("dropoff_time"),
        pl.col("PULocationID").alias("pickup_loc"),
        pl.col("DOLocationID").alias("dropoff_loc"),
        pl.col("trip_distance"),
        pl.col("fare_amount").alias("fare"),
        pl.col("total_amount"),
        pl.col("congestion_surcharge").fill_null(0.0),
        pl.col("VendorID")
    ])
    
    # 3. CONCATENATE
    # Now that columns match perfectly, we stack them.
    return pl.concat([q_yellow, q_green])