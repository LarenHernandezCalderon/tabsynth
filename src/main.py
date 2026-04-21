import os
import pandas as pd
from dotenv import load_dotenv
from analyzer.dataset_analyzer import DatasetAnalyzer

load_dotenv()

df      = pd.read_csv("../data/titanic.csv")
profile = DatasetAnalyzer(df, openai_api_key=os.getenv("OPENAI_API_KEY")).analyze()