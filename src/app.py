import streamlit as st
import pandas as pd
import pickle
from transform_data import Transform

st.title('Potential Churn Patients')

t = Transform()
total = t.pay_transform('../data/raw/payment.csv', '../data/raw/claims.csv')
patient = t.patient_transform('../data/raw/appt.csv', '../data/raw/patient.csv')
merged = patient.merge(total)
for_model, contact = t.data_split(merged)

@st.cache
def load_data(link_data, drop_columns):
    """
    Loads data from source and parses into a test set for predictions

    Parameters:
        link_data: original dataframe to link patients with after making predictions on churn
        drop_columns: list of columns to be dropped from raw dataset for use with model predictions

    Returns:
        Original dataset to map back to patient contact info and test set to be used for making predictions
    """

    #link_data = orig_data[orig_data['Recency'].between(start_day, end_day)].reset_index(drop=True)
    test_data = link_data.drop(drop_columns, axis=1)
    return link_data, test_data

def load_model(filepath):
    """
    Loads Picked model for use with predictions

    Parameters:
        Filepath to pickled model

    Returns:
        Pickled model for use with predictions on data
    """
    
    with open(filepath, 'rb') as file:
        Pickled_Model = pickle.load(file)
    return Pickled_Model

def priority_list(original_df, predicted_probas, thresh=0.48, num_patients=20):
    """
    Create patient prioritized contact list to prevent churn

    Parameters:
        original df:      df to map back to patient contact info
        predicted_probas: predicted probabilities for each patients to be sorted highest to lowest for churn (must be 2d array for binary class)
        thresh:           threshold setting for capturing predicted churn above a certain threshold, default = 0.48
        num_patients:     user input feature to allow setting the number of patients that the receptionist wants to contact each day

    Returns:
        Prioritized list of potential churn patients for staff to take action on (includes patient contact info)
    """
    
    preds = pd.DataFrame(predicted_probas)
    churns = preds[preds[1] >= thresh].loc[:,1].sort_values(ascending=False)
    priority_patients = churns.iloc[:num_patients]
    indices = priority_patients.index.tolist()
    patients = original_df.loc[indices, ['PatNum', 'FName', 'Tenure', 'Frequency', 'Recency']]
    #patients.insert(5, 'Risk Factor', round(priority_patients,1))
    patients.columns = ['PatNum', 'First Name', 'Tenure', '#_of_Visits', 'Last Visit (days)']#, 'Risk Factor']
    patients.index = patients.reset_index(drop=True).index + 1
    return patients


dropcols = ['PatNum', 'FName',  'Recency']
#load data from source and get create original df and data for getting predictions
link_data, test = load_data(link_data=for_model, drop_columns=dropcols)

# load pretrained model and make predictions
model = load_model('bestLRmodel.pkl')
predict_probas = model.predict_proba(test)

#allow use to input threshold value for prediction probabilities
num_patients = st.text_input(label='# of Patients to Contact', value=10, max_chars=None, key=1, type='default')
df = priority_list(link_data, predict_probas, num_patients=int(num_patients))
#st.title(f'Total # of Patients: {len(df)}')

#display patient data on screen
st.table(df)

st.title('Prioritized Contact List')
num_patients2 = st.text_input(label='# of Patients to Contact', value=10, max_chars=None, key=2, type='default')
contact_df = t.contact_transform(contact)
contact_df.index = contact_df.reset_index(drop=True).index + 1
st.table(contact_df.iloc[:int(num_patients2)])

