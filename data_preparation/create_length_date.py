import pandas as pd
import numpy as np

def create_length_date(df):

    df_temp = df.copy()
    
    # Create Measurement Date
    measurement_date = '2023-07-01'
    measurement_date = pd.to_datetime(measurement_date)

    # Calculate employment lenght
    df_temp['tenure'] = np.where(
        df_temp['Datum_uit_dienst'].notna(),
        df_temp['Datum_uit_dienst'] - df_temp['Datum_in_dienst'],
        measurement_date - df_temp['Datum_in_dienst']
    )

    df_temp['functie_lengte'] = np.where(
        df_temp['Einddatum_functie'].notna(),
        df_temp['Einddatum_functie'] - df_temp['Begindatum_functie'],
        measurement_date - df_temp['Begindatum_functie']
    )

    df_temp['leeftijd'] = measurement_date - df_temp['Geboortedatum']

    df_temp['jaren_tot_aow'] = df_temp['AOW-datum'] - measurement_date

    # Convert tenure to number of days
    df_temp['tenure'] = df_temp['tenure'].dt.days / 365.25
    df_temp['functie_lengte'] = df_temp['functie_lengte'].dt.days / 365.25
    df_temp['leeftijd'] = df_temp['leeftijd'].dt.days / 365.25
    df_temp['jaren_tot_aow'] = df_temp['jaren_tot_aow'].dt.days / 365.25

    df_temp = df_temp.drop(columns=['Begindatum_contract', 'Einddatum_contract', 'Datum_uit_dienst', 'Datum_in_dienst', 'Einddatum_functie', 'Begindatum_functie', 'Geboortedatum', 'AOW-datum'])
    
    return df_temp

    