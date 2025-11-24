import pandas as pd
import requests
import logging
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LocationConfig:
    """Config object to keep variables out of logic."""
    name: str
    lat: float
    lon: float

class WeatherIngestionPipeline:
    
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, config: LocationConfig):
        self.config = config

    def fetch_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
       
        params = {
            "latitude": self.config.lat,
            "longitude": self.config.lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_max", "precipitation_sum", "snowfall_sum"],
            "timezone": "America/New_York",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch"
        }

        try:
            logger.info(f"Initiating API request for {self.config.name}...")
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data['daily'])
            df['time'] = pd.to_datetime(df['time'])
            
            logger.info(f"Successfully ingested {len(df)} records.")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"API Connection Failed: {e}")
            return None

    def process_winter_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters for winter months and flags severe weather events."""
        if df is None or df.empty:
            logger.warning("No data to process.")
            return pd.DataFrame()

        # Filter: Dec (12), Jan (1), Feb (2), Mar (3)
        winter_df = df[df['time'].dt.month.isin([12, 1, 2, 3])].copy()
        
        # Vectorized calculation for risk flags (Faster than iteration)
        winter_df['risk_level'] = 'LOW'
        winter_df.loc[winter_df['snowfall_sum'] > 2.0, 'risk_level'] = 'HIGH'
        
        high_risk_count = len(winter_df[winter_df['risk_level'] == 'HIGH'])
        logger.info(f"Processed Winter Data. High Risk Events Identified: {high_risk_count}")
        
        return winter_df

    def run(self, start: str, end: str, output_file: str):
        """Orchestrator function."""
        raw_data = self.fetch_data(start, end)
        if raw_data is not None:
            processed_data = self.process_winter_risk(raw_data)
            processed_data.to_csv(output_file, index=False)
            logger.info(f"Pipeline complete. Data exported to {output_file}")

if __name__ == "__main__":
    # Configuration is now separated from logic
    target_loc = LocationConfig(name="Blacksburg_VA", lat=37.229, lon=-80.413)
    
    # Initialize and Run
    pipeline = WeatherIngestionPipeline(target_loc)
    pipeline.run(start="1990-01-01", end="2023-12-31", output_file="blacksburg_winter_risk.csv")
