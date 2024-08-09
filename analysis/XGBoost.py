# Loading packages

from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_preparation.clean_data import clean_data
from data_preparation.create_length_date import create_length_date
from data_preparation.data_prep import data_prep
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



def xgboost_turnover(df):
    identifier = df['Medewerker_code']

    # Separate features and target variable, excluding 'Medewerker_code'
    features = df.drop(columns=['Medewerker_code', 'tenure', 'Status_dienstverband_Uit dienst'])
    target = df['tenure']

    # Split the dataset
    df_left = df[df['Status_dienstverband_Uit dienst'] == True]
    df_current = df[df['Status_dienstverband_Uit dienst'] == False]

    # Further split df_left into training and validation sets
    X_train_full = df_left.drop(columns=['Medewerker_code', 'tenure', 'Status_dienstverband_Uit dienst'])
    y_train_full = df_left['tenure']
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # Features for the testing set (current employees)
    X_test = df_current.drop(columns=['Medewerker_code', 'tenure', 'Status_dienstverband_Uit dienst'])

    # Define model parameters
    params = {
        "objective": "reg:linear",
        "max_depth": 4,
        "eta": 0.1,  # Learning rate
        "subsample": 0.8,  # Subsample ratio of the training instances
        "colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree
        "seed": 42  # Random seed
    }

    # Initialize and train the XGBoost regressor
    model = XGBRegressor(**params, n_estimators=10)
    model.fit(X_train, y_train)

    # Predict the tenure of current employees
    predictions = model.predict(X_test)

    # Add predictions and 'Medewerker_code' back to the dataframe for current employees
    df_current['Predicted_Tenure'] = predictions
    df_current['Medewerker_code'] = identifier[df_current.index]

    # Predict on the validation set and calculate performance
    val_predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, val_predictions)
    print(f"Mean Squared Error on validation set: {mse}")

    # Plot feature importance
    plot_importance(model)
    plt.show()

    # Display the dataframe with predictions and 'Medewerker_code'
    print(df_current[['Medewerker_code', 'Predicted_Tenure']])  # Adjust according to your dataframe columns

    # Evaluate the model (optional, if you have a validation set)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Mean Squared Error: {mse}")
    df_current.to_excel(r'data/processed_dataframe.xlsx')
    


def merge_dataframes():
    df_raw = pd.read_excel(r'data/dataset.xlsx')
    df_current = pd.read_excel(r'data/processed_dataframe.xlsx')
    
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
    # Change date columns to datetime 
    for column in date_cols:
        df_raw[column] = pd.to_datetime(df_raw[column], utc=True, errors='coerce').dt.tz_localize(None)


    df_raw = df_raw.merge(df_current[['Medewerker_code', 'Predicted_Tenure']], how='left', left_on='Medewerker_code', right_on='Medewerker_code')
    df_raw
    
        
    # Calculate the possible leave date by adding the predicted tenure (in years) to the contract starting day
    df_raw['Possible_Leave_Date'] = df_raw['Begindatum_contract'] + pd.to_timedelta(df_raw['Predicted_Tenure'] * 365, unit='D')

    # Display the dataframe with predictions
    print(df_raw[['Medewerker_code', 'Begindatum_contract', 'Predicted_Tenure', 'Possible_Leave_Date']])  # Adjust according to your dataframe columns
    
    

    # Ensure 'Possible_Leave_Date' is a datetime object
    df_raw['Possible_Leave_Date'] = pd.to_datetime(df_raw['Possible_Leave_Date'])

    # Sort the dataframe by 'Possible_Leave_Date' in descending order
    df_raw = df_raw.sort_values(by='Possible_Leave_Date', ascending=False)

    # Ensure EmployeeID is a string for plotting purposes
    df_raw['EmployeeID'] = df_raw['Medewerker_code'].astype(str)

    # Get today's date
    today = pd.to_datetime('today').normalize()

    # Plot
    plt.figure(figsize=(12, 8))

    # Plot the possible leave date
    plt.barh(df_raw['EmployeeID'], df_raw['Possible_Leave_Date'], color='skyblue', edgecolor='black')

    # Add a dotted line 2 years from today
    two_years_from_today = today + pd.DateOffset(years=2)
    plt.axvline(two_years_from_today, color='red', linestyle='dotted', linewidth=2, label='2 years from today')

    # Customize the plot
    plt.xlabel('Date')
    plt.ylabel('Employee Code')
    plt.title('Employees Possible Leave Date')
    plt.xticks(rotation=45)
    plt.legend()

    # Set the x-axis limits to start from today
    plt.xlim([today, df_raw['Possible_Leave_Date'].max() + pd.DateOffset(months=6)])

    # Invert y-axis to show descending order
    plt.gca().invert_yaxis()

    # Display the plot
    plt.tight_layout()
    plt.show()
    plt.savefig('plots/remaining_years.png')

