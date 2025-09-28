import pandas as pd


# Base URL for raw GitHub repo (public repo so should work for anyone)
BASE_URL = "https://github.com/siddharthyashkukreja-cloud/QuantInvesting/blob/35528ac1dde5d2d873e1adfd0864381326d3b768"
# Loading datasets directly from GitHub
df = pd.read_csv(f"{BASE_URL}/predictor_data.csv")
