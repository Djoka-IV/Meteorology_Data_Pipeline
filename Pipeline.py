# -*- coding: utf-8 -*-
"""
Operational Weather Risk Pipeline
Author: Djordje Ivosevic
"""

import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configure Logging (Outputs to console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class LocationConfig:
    """Immutable configuration for target location parameters."""
    name: str
    lat: float
    lon: float

class WeatherIngestionPipeline:
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, config: LocationConfig):
        self.config = config

    def fetch_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Retrieves historical weather data from the Open-Meteo Archive API.
        Handles connection timeouts and basic data sanitization.
        """
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
            
            # Data imputation: Zero-fill NaNs to ensure downstream model stability
            df = df.fillna(0)
            
            logger.info(f"Successfully ingested {len(df)} records.")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"API Connection Failed: {e}")
            return None

    def process_winter_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters dataset for seasonal relevance and applies heuristic labeling.
        Generates synthetic ground truth for model training based on operational thresholds.
        """
        if df is None or df.empty:
            logger.warning("No data available for processing.")
            return pd.DataFrame()

        # Filter: Winter Months (Dec, Jan, Feb, Mar)
        winter_df = df[df['time'].dt.month.isin([12, 1, 2, 3])].copy()
        
        # Operational Logic: Generate binary risk labels
        # Threshold: > 2.0 inches of snowfall constitutes a High Risk event (1)
        winter_df['risk_label'] = 0 
        winter_df.loc[winter_df['snowfall_sum'] > 2.0, 'risk_label'] = 1 
        
        return winter_df

    def train_risk_model(self, df: pd.DataFrame):
        """
         trains a Logistic Regression classifier on the processed meteorological features.
        """
        logger.info("Initializing ML training sequence...")

        # Feature Engineering
        features = df[['temperature_2m_max', 'precipitation_sum', 'snowfall_sum']]
        target = df['risk_label']

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Model Training
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Validation
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        logger.info(f"Model Training Complete. Validation Accuracy: {accuracy:.4f}")
        
        return model

    def run(self, start: str, end: str, output_file: str):
        """
        Orchestration method to execute the full ELT and ML pipeline.
        """
        logger.info("Pipeline execution started.")
        
        # 1. Ingest
        raw_data = self.fetch_data(start, end)
        
        if raw_data is not None:
            # 2. Transform & Label
            processed_data = self.process_winter_risk(raw_data)
            
            # 3. Model Training
            if not processed_data.empty:
                self.train_risk_model(processed_data)
            else:
                logger.warning("Processed dataset is empty. Skipping model training.")
            
            # 4. Export
            processed_data.to_csv(output_file, index=False)
            logger.info(f"Pipeline complete. Data serialized to {output_file}")

if __name__ == "__main__":
    # Initialize Configuration
    target_loc = LocationConfig(name="Blacksburg_VA", lat=37.229, lon=-80.413)
    
    # Execute Pipeline
    pipeline = WeatherIngestionPipeline(target_loc)
    pipeline.run(start="1990-01-01", end="2023-12-31", output_file="blacksburg_winter_risk.csv")
