import pandas as pd

def data_prep(df):
    
    df_temp = df.copy()
    
    categoric_columns = [
        'Type_contract_omschrijving',
        'Dienstbetrekking_omschrijving',
        'Status_dienstverband',
        'Geslacht_omschrijving',
        'Is_leidinggevende'
        ]

    df_temp = pd.get_dummies(
        df_temp, 
        columns=categoric_columns, 
        drop_first=True
        )
    
    #df_temp['#FULLTIME_SALARY'] = df_temp['#FULLTIME_SALARY'].fillna(df_temp['#FULLTIME_SALARY']).mean()


    return df_temp