OPERATIONAL WEATHER RISK PIPELINE
Author: Djordje Ivosevic
OVERVIEW
This project is an automated ETL and Machine Learning pipeline designed to ingest historical climatological data and predict high-impact operational risk events. It retrieves 30+ years of daily weather data, normalizes it for analysis, and trains a Logistic Regression classifier (Scikit-Learn) to identify severe weather conditions that could disrupt supply chains.
TECHNICAL HIGHLIGHTS
Machine Learning Integration: Implements a Scikit-Learn workflow to train predictive models on historical meteorological features.
Object-Oriented Design: Uses a modular class structure with separated configuration (Dataclasses) to ensure code maintainability.
Efficient Processing: Utilizes vectorized Pandas and NumPy operations to handle large time-series datasets with minimal latency.
Production Standards: Features robust error handling, type hinting, and logging to manage API timeouts and data discontinuities.
DEPENDENCIES
The project requires the following libraries:
pandas (Data manipulation)
requests (API ingestion)
numpy (Numerical operations)
scikit-learn (Model training)
USAGE
Install Dependencies:
pip install -r requirements.txt
Run the Pipeline:
python main.py
OUTPUT
The script will output model validation metrics (accuracy) to the console and generate a CSV file containing the processed risk assessment dataset.
Use Arrow Up and Arrow Down to select a turn, Enter to jump to it, and Escape to return to the chat.
