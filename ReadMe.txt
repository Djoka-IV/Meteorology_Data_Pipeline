Operational Weather Risk Pipeline
  Overview
    This project is an automated ETL (Extract, Transform, Load) pipeline designed to ingest historical climatological data and identify high-impact operational risk events.
    It automates the retrieval of 30+ years of daily weather data, filters for specific operational seasons (Winter), and applies business logic thresholds to flag days that would cause supply chain or safety disruptions (e.g., Heavy Snowfall).
 
  Technical Highlights
    Object-Oriented Design: Uses a class-based structure with separated configuration logic (Dataclasses) for modularity.
    Production Standards: Implements proper Type Hinting, Logging, and Error Handling to ensure stability.
    Efficient Processing: Utilizes vectorized Pandas and NumPy operations for fast data manipulation.
    API Integration: Manages external connections to the Open-Meteo Archive API with timeout protection.

  Dependencies
    The project relies on the following key libraries (versions specified in requirements.txt):
    pandas (>=2.0.0)
    requests (>=2.31.0)
    numpy (>=1.24.0)

  Usage
    Install Dependencies:
      pip install -r requirements.txt

  Run the Pipeline:
    python main.py

  Output:
    The script will generate a CSV file containing the processed risk assessment for the target location.

Author: Djordje Ivosevic
Purpose: Portfolio demonstration of Python automation for meteorological risk assessment.
