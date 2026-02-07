import polars as pl
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime
from ingest import ingest_and_unify # Import Phase 1 script

OUTPUT_DIR = "./outputs"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Manhattan Zone IDs (South of 60th St)
CONGESTION_ZONES = [
    12, 13, 43, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114, 116, 120, 
    125, 137, 140, 141, 142, 143, 144, 148, 151, 158, 161, 162, 163, 164, 166, 
    170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 
    243, 244, 246, 249, 261, 262, 263
]

def fetch_weather():
    """Fetches 2025 daily rain data for Central Park"""
    print("☁️ Fetching Weather Data...")
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 40.7831, "longitude": -73.9712, 
        "start_date": "2025-01-01", "end_date": "2025-12-31",
        "daily": "precipitation_sum", "timezone": "America/New_York"
    }
    responses = openmeteo.weather_api(url, params=params)
    daily = responses[0].Daily()
    
    # FIX: Use datetime_range and explicitly cast to strict 'Date' type
    return pl.DataFrame({
        "date": pl.datetime_range(
            datetime(2025,1,1), 
            datetime(2025,12,31), 
            "1d", 
            eager=True
        ).cast(pl.Date),  # <--- This cast fixes the Join Error
        "precipitation_mm": daily.Variables(0).ValuesAsNumpy()
    })

def run_pipeline():
    # 1. Ingest Data (LazyFrame)
    q = ingest_and_unify()
    
    # 2. Feature Engineering
    q = q.with_columns([
        (pl.col("dropoff_time") - pl.col("pickup_time")).dt.total_minutes().alias("duration_min"),
        pl.col("pickup_time").dt.date().alias("date"),
        pl.col("pickup_time").dt.hour().alias("hour"),
        pl.col("pickup_time").dt.weekday().alias("weekday")
    ])
    
    q = q.with_columns(
        (pl.col("trip_distance") / (pl.col("duration_min") / 60)).alias("speed_mph")
    )

    # 3. The Ghost Trip Audit (Filter Dirty Data)
    print("👻 Auditing Ghost Trips...")
    ghost_criteria = (
        (pl.col("speed_mph") > 65) | 
        ((pl.col("duration_min") < 1) & (pl.col("fare") > 20)) |
        ((pl.col("trip_distance") == 0) & (pl.col("fare") > 0))
    )
    
    # Save Ghost Stats separate from clean data
    ghost_stats = q.filter(ghost_criteria).group_by("VendorID").len()
    
    # Clean Data
    q_clean = q.filter(~ghost_criteria)

    # 4. Aggregations (Reduce Data Size)
    
    # A. Leakage (Starts Outside -> Ends Inside -> No Surcharge)
    print("🔍 Calculating Leakage...")
    leakage_agg = q_clean.filter(
        (~pl.col("pickup_loc").is_in(CONGESTION_ZONES)) & 
        (pl.col("dropoff_loc").is_in(CONGESTION_ZONES))
    ).group_by("pickup_loc").agg([
        pl.len().alias("total_trips"),
        (pl.col("congestion_surcharge") == 0).sum().alias("missing_surcharge_count")
    ]).sort("missing_surcharge_count", descending=True).limit(10)

    # B. Velocity Heatmap
    print("🚀 Calculating Velocity...")
    velocity_agg = q_clean.filter(
        pl.col("pickup_loc").is_in(CONGESTION_ZONES)
    ).group_by(["weekday", "hour"]).agg(
        pl.col("speed_mph").mean().alias("avg_speed")
    )

    # C. Economics (Tips)
    print("💰 Calculating Economics...")
    economics_agg = q_clean.with_columns(
        pl.col("date").dt.truncate("1mo").alias("month")
    ).group_by("month").agg([
        pl.col("congestion_surcharge").mean().alias("avg_surcharge"),
        # Tip % calculation (Tip / Fare)
        (pl.col("total_amount") - pl.col("fare")).mean().alias("avg_tip_amt") # Approx
    ])
    
    # D. Daily Trip Counts (For Weather Join)
    # --- THIS WAS THE FIX ---
    daily_counts = q_clean.group_by("date").agg(
        pl.len().alias("trip_count")
    )

    # 5. EXECUTE (Collect & Save)
    print("💾 Saving Aggregated Outputs (This may take a moment)...")
    
    # Collect dataframes (executes the lazy plan)
    ghost_stats.collect().write_csv(f"{OUTPUT_DIR}/ghost_audit.csv")
    leakage_agg.collect().write_csv(f"{OUTPUT_DIR}/leakage_audit.csv")
    velocity_agg.collect().write_csv(f"{OUTPUT_DIR}/velocity_heatmap.csv")
    economics_agg.collect().write_csv(f"{OUTPUT_DIR}/economics.csv")
    
    # Weather Join (In Memory)
    weather_df = fetch_weather()
    trips_df = daily_counts.collect() # Collect trips before joining
    
    # Inner join trips with weather
    weather_df.join(trips_df, on="date", how="inner").write_csv(f"{OUTPUT_DIR}/weather_elasticity.csv")
    
    print("✅ Pipeline Complete.")

if __name__ == "__main__":
    run_pipeline()