import streamlit as st
import pandas as pd
import pickle
from transform_data import Transform

st.title('Potential Churn Patients')

t = Transform()
total = t.pay_transform('../data/raw/payment.csv', '../data/raw/claims.csv')
patient = t.patient_transform('../data/raw/appt.csv', '../data/raw/patient.csv')
merged = patient.merge(total)
model, contact = t.data_split(merged)

@st.cache
def load_data(model_data, drop_columns, start_day=150, end_day=399):
    """
    Loads data from source and parses into a test set for predictions

    :param: filepath to data file
    :param: day_range = range of days to select churn window
    :param: drop_columns = list of columns to be dropped from raw dataset for use with model predictions
    returns: original dataset to map back to patient contact info and test set for predictions
    """
    
    orig_data = model_data
    link_data = orig_data[orig_data['Recency'].between(start_day, end_day)].reset_index(drop=True)
    test_data = link_data.drop(drop_columns, axis=1)
    return link_data, test_data

def load_model(filepath):
    """
    Loads Picked model for use with predictions

    params: filepath to pickled model
    returns: pickled model for use with predictions on data
    """
    
    with open(filepath, 'rb') as file:
        Pickled_Model = pickle.load(file)
    return Pickled_Model

def priority_list(original_df, predicted_probas, thresh=75):
    """
    Create patient prioritized contact list to prevent churn

    :param: original df: df to map back to patient contact info
    :param predicted_probas: predicted probabilities for each patients to be sorted highest to lowest for churn (must be 2d array for binary class)
    :param thresh: threshold setting for capturing predicted churn above a certain threshold, default = 70
    """
    
    preds = pd.DataFrame(predicted_probas) * 100
    churns = preds[preds[1] >= thresh].loc[:,1].sort_values(ascending=False)
    patients = original_df.loc[churns.index, :].loc[:, ['PatNum', 'FName', 'Tenure', 'Frequency', 'Recency']]
    patients.insert(5, 'Risk Factor', round(churns,1))
    patients.columns = ['PatNum', 'First Name', 'Tenure', '#_of_Visits', 'Last Visit (days)', 'Risk Factor']
    patients.index = patients.reset_index(drop=True).index + 1
    return patients


dropcols = ['PatNum', 'FName',  'Recency']
#load data from source and get create original df and data for getting predictions
link_data, test = load_data(model_data=model, drop_columns=dropcols)

# load pretrained model and make predictions
model = load_model('bestLRmodel.pkl')
predictions = model.predict_proba(test)

#allow use to input threshold value for prediction probabilities
thresh = st.text_input(label='Risk Factor Threshold', value=75, max_chars=None, key=None, type='default')
df = priority_list(link_data, predictions, thresh=float(thresh))
st.title(f'Total # of Patients: {len(df)}')

#display patient data on screen
st.dataframe(df)

#st.title('Prioritized Contact List')


