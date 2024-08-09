import pandas as pd
import numpy as np

def clean_data(df):
    
    # make a copy of the dataset
    df_temp = df.copy()
    
    # Define the relevant columns to analyze
    relevant_columns = [
    'Medewerker_code', 
    'Begindatum_contract', 
    'Einddatum_contract', 
    'Type_contract_omschrijving', 
    'Dienstbetrekking_omschrijving', 
    'Datum_in_dienst', 
    'Datum_uit_dienst',  
    'Status_dienstverband', 
    'Geboortedatum', 
    'Geslacht_omschrijving', 
    'AOW-datum', 
    'Ziektedagen_gewogen_va_2017', 
    'Aantal_ziekmeldingen_va_2017', 
    'Begindatum_functie', 
    'Einddatum_functie', 
    'Is_leidinggevende', 
    '#FULLTIME_SALARY'
    ]
    
    # select the relevant columns
    df_temp = df_temp.loc[:, relevant_columns]
    
    # Define Date Columns
    date_cols = [
        'Begindatum_contract',
        'Einddatum_contract',
        'Datum_in_dienst',
        'Datum_uit_dienst',
        'Geboortedatum',
        'AOW-datum',
        'Begindatum_functie',
        'Einddatum_functie'
        ]

    # Define Cateogric Columns
    categoric_columns = [
        'Type_contract_omschrijving',
        'Dienstbetrekking_omschrijving',
        'Status_dienstverband',
        'Geslacht_omschrijving',
        'Is_leidinggevende'
        ]

    # Change date columns to datetime 
    for column in date_cols:
        df_temp[column] = pd.to_datetime(df_temp[column], utc=True, errors='coerce').dt.tz_localize(None)

    # Change categoric columns to categoric type
    for column in categoric_columns:
        df_temp[column] = df_temp[column].astype('category')
    
    # Replace "-" values with Nan
    df_temp.replace("-", np.nan, inplace=True)
    
    # Drop duplicate employees
    df_temp = df_temp.drop_duplicates(subset=['Medewerker_code'], keep='last')
    
    # change irregular calues such as comma to decimal point and erase $ sign
    df_temp['#FULLTIME_SALARY'] = df_temp['#FULLTIME_SALARY'].str.replace('$', '').str.replace(',', '').astype('float')
    df_temp['Ziektedagen_gewogen_va_2017'] = df_temp['Ziektedagen_gewogen_va_2017'].str.replace(',', '.').astype('float')
    
    df_temp = df_temp.rename(columns={'#FULLTIME_SALARY': 'Salary'})
    
    return df_temp
