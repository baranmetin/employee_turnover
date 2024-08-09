import pandas as pd
import numpy as np
from data_preparation.clean_data import clean_data
from data_preparation.create_length_date import create_length_date
from data_preparation.data_prep import data_prep
from analysis.cox_ph import cox_ph_survival
from analysis.XGBoost import xgboost_turnover, merge_dataframes


path = r'data\dataset.xlsx'

df_raw = pd.read_excel(path)

df_clean = clean_data(df_raw)
df_clean = create_length_date(df_clean)
df_clean = data_prep(df_clean)

df = df_clean.copy()

def main():
    cox_ph_survival(df)
    #xgboost_turnover(df)
    #merge_dataframes()
    
if __name__ == "__main__":
    main()