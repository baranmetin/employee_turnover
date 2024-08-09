from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import k_fold_cross_validation, find_best_parametric_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os

def cox_ph_survival(df):
    
    df_temp = df.copy()
    # Ensure data types are correct
    df_temp = df_temp.apply(pd.to_numeric, errors='coerce')
    print(df_temp.dtypes)

    
    low_variance_cols = ['Salary', 'Geslacht_omschrijving_Non-binair']
    df_temp = df_temp.drop(columns=low_variance_cols)
    
    # Tenure transformation (e.g., log transformation)
    #df_temp['log_tenure'] = np.log1p(df_temp['tenure'])
    
    cph = CoxPHFitter()
    cph.fit(df_temp, duration_col = 'tenure', event_col = 'Status_dienstverband_Uit dienst')
    
    print(cph.print_summary())
    
    # First plot
    fig1, ax1 = plt.subplots(figsize=(12, 8))  # Adjust the width and height as needed
    cph.plot(ax=ax1)
    ax1.set_title("Cox Proportional Hazards Model")
    plt.tight_layout()
    #plt.savefig("plots/cox_ph_survival_plot.png")
    #plt.show()
    
    # Second plot: Partial effects on outcome
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    cph.plot_partial_effects_on_outcome(covariates="jaren_tot_aow", values=[0, 10, 20, 30], ax=ax2)
    ax2.set_title("Partial Effects on Outcome")
    plt.tight_layout()
    #plt.savefig("plots/partial_effects_plot.png")
    #plt.show()


     # predictions

    last_obs = df_temp.apply(lambda row: row["tenure"] if row["Status_dienstverband_Uit dienst"] == 0 else 0, axis=1)
    last_obs = pd.to_numeric(last_obs, errors='coerce')

    # predict median remaining life 
    remaining_life = cph.predict_median(df_temp, conditional_after=last_obs) # here is an error; 
                                                                                # TypeError: Cannot cast array data from dtype('O') 
                                                                                # to dtype('float64') according to the rule 'safe'
                                                                                # , conditional_after=last_obs
    remaining_life.name = "remaining_life"

    # create output
    
    # Predict remaining life
    try:
        remaining_life = cph.predict_median(df_temp, conditional_after=last_obs)
        remaining_life.name = "remaining_life"
    except TypeError as e:
        print("Error during prediction:", e)

    # Create output
    out = df_temp.copy()
    for p in np.arange(0.9, 0.5, -0.1):
        try:
            remaining_life = cph.predict_percentile(df_temp, p=p, conditional_after=last_obs)
            remaining_life.name = "remaining_jaren_p_{:.0f}".format(p*100)
            out = pd.merge(out, remaining_life, left_index=True, right_index=True)
        except TypeError as e:
            print("Error during prediction at percentile {}: {}".format(p, e))

    # Inspect output
    print(out.head())
    
    #out = df.copy()
    #for p in np.arange(0.9, 0.5, -0.1):
    #    remaining_life = cph.predict_percentile(df_temp, p=p, conditional_after=last_obs) #, conditional_after=last_obs)
    #    remaining_life.name = "remaining_jaren_p_{:.0f}".format(p*100)
    #    out = pd.merge(out, remaining_life, left_index=True, right_index=True)

    # save output only non-observed resigns
    out[out["Status_dienstverband_Uit dienst"] == 0].to_csv(os.path.join("reports", "predictions_turnover.csv"))