import pandas as pd
import numpy as np


class Transform:
    def __init__(self):
        pass

    def pay_transform(self, pay_filepath, claims_filepath):
        """
        Transforms raw data into grouped values of all patient payments made, both out of pocket and insurance

        :param filepaths: paths to respective pay and claims tables
        :return : merged table with Total of all patient payments over course of entire patient life
        """

        # clean and transform pay table
        pay = pd.read_csv(pay_filepath, usecols=['PayDate', 'PatNum', 'PayAmt'])
        pay = pay[pay['PayDate'] != '2020-12-22']
        grouped_pay = pay.groupby("PatNum", as_index=False)['PayAmt'].sum()

        # clean and transform claims table
        claims = pd.read_csv(claims_filepath, engine='python', error_bad_lines=False,
                             usecols=['PatNum', 'DateReceived', 'InsPayAmt'])
        claims = claims[claims['DateReceived'] != '0001-01-01']
        claims.loc[17482, 'InsPayAmt'] = 754
        claims.drop('DateReceived', axis=1, inplace=True)
        grouped_claims = claims.groupby('PatNum', as_index=False).sum()

        # merge tables and create "TOTAL" for further use
        merged = grouped_claims.merge(grouped_pay)
        merged['Total'] = merged['InsPayAmt'] + merged['PayAmt']
        merged = merged.loc[:, ['PatNum', 'Total']]

        #merged.to_csv('../data/model/total.csv', index=False)
        return merged

    def patient_transform(self, appt_filepath, pat_filepath):
        """
        Transforms raw data into patient df for use in predictive modeling

        :params: filepaths to appt table, patient table, and total payments table
        """

        patient_cols = ['FName', 'PatNum', 'Birthdate', 'Gender', 'EstBalance', 'InsEst', 'HasIns', 'DateFirstVisit']
        appt = pd.read_csv(appt_filepath, usecols=['PatNum', 'ProvNum', 'AptStatus', 'AptDateTime'])
        pat = pd.read_csv(pat_filepath, usecols=patient_cols)

        #drop cancelled appointments
        appt = appt[appt['AptStatus'] != 5]
        appt.drop('AptStatus', axis=1, inplace=True)

        #drop bad date entries (only 2)
        appt = appt[appt['AptDateTime'] != '0001-01-01 00:00:00']

        #drop fake patients
        appt = appt[~appt['PatNum'].isin([3645, 5686, 3391, 2, 5557, 2661])]

        #remove time from date/time column
        appt['AptDateTime'] = appt['AptDateTime'].str[:10]
        appt['AptDateTime'] = pd.to_datetime(appt['AptDateTime'])

        # remove incorrect birthdates and transform date columns
        pat['Birthdate'] = np.where(pat['Birthdate'] == '0001-01-01', np.nan, pat['Birthdate'])
        pat['Birthdate'] = pd.to_datetime(pat['Birthdate'])

        # create age column and fill nan's with mean age
        now = pd.to_datetime('now')
        pat['age'] = (now - pat['Birthdate']).astype('<m8[Y]')
        pat.age.fillna(pat.age.mean(), inplace=True)
        pat.drop('Birthdate', axis=1, inplace=True)

        # drop inactive patients and transform HasIns col
        pat = pat[pat['DateFirstVisit'] != '0001-01-01']
        pat['DateFirstVisit'] = pd.to_datetime(pat['DateFirstVisit'])
        pat['HasIns'] = np.where(pat['HasIns'] == 'I', 1, 0)

        # create Provider dictionary to create binary "seen by hygenist X" columns
        provider_dict = {k: (appt[appt['ProvNum'] == k]['PatNum']).unique() for k in [1, 2, 6, 7, 10, 15]}

        pat['seen_by_1'] = np.where(pat['PatNum'].isin(provider_dict[1]), 1, 0)
        pat['seen_by_2'] = np.where(pat['PatNum'].isin(provider_dict[2]), 1, 0)
        pat['seen_by_6'] = np.where(pat['PatNum'].isin(provider_dict[6]), 1, 0)
        pat['seen_by_7'] = np.where(pat['PatNum'].isin(provider_dict[7]), 1, 0)
        pat['seen_by_10'] = np.where(pat['PatNum'].isin(provider_dict[10]), 1, 0)
        pat['seen_by_15'] = np.where(pat['PatNum'].isin(provider_dict[15]), 1, 0)

        #create new Frequency, Last Visit, Tenure and Recency columns
        grouped = appt.groupby('PatNum')['AptDateTime'].agg(['count', 'max']).reset_index()
        grouped.columns = ['PatNum', 'Frequency', 'Last Visit']
        merged = pat.merge(grouped)
        merged['Tenure'] = merged['Tenure'] = (merged['Last Visit'] - merged['DateFirstVisit']).dt.days
        merged['Recency'] = (pd.to_datetime('now') - merged['Last Visit']).dt.days

        #drop all time based columns
        merged.drop(['DateFirstVisit', 'Last Visit'], axis=1, inplace=True)

        return merged

    def data_split(self, dataframe, churn_end=400, contact_end=720):
        """
        Splits dataframe into two sets, one for making churn predictions and one for creating prioritized contact list

        :params dataframe: merged pandas DF from previous steps
        :param churn_end: allows user to select defined churn timeframe, default is 400 days from last visit
        :param contact_end: allows user to select end of timeframe for contact list, default is 720 days from last visit (2 years)
        :returns: two pandas DFs
        """

        for_model = dataframe[dataframe['Recency'].between(150, churn_end)]
        contact_list = dataframe[dataframe['Recency'].between(400, contact_end)]

        return for_model, contact_list

if __name__ == '__main__':
    t = Transform()
    total = t.pay_transform('../data/raw/payment.csv', '../data/raw/claims.csv')
    patient = t.patient_transform('../data/raw/appt.csv', '../data/raw/patient.csv')
    merged = patient.merge(total)
    model, contact = t.data_split(merged)
    model.to_csv('../data/model/for_model_preds.csv', index=False)
    contact.to_csv('../data/model/for_contact_list.csv', index=False)
