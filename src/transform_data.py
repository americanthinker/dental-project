import pandas as pd
import numpy as np


class Transform:
    def __init__(self):
        pass

    def pay_transform(self, pay_filepath, claims_filepath):
        """
        Transforms raw data into grouped values of all patient payments made, both out of pocket and insurance

        Parameters:
             pay_filepath: path to payment table
             claims_filepath: path to claims table
        Returns:
            Merged table with Total of all patient payments over course of entire patient life
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

        Parameters:
             appt_filepath: filepath to appointment table
             pat_filepath: filepath to patient table

        Returns:
            Merged DataFrame of appt and patient tables for use with model for predictions and contact list
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

    def data_split(self, dataframe, churn_begin=150, churn_end=399, contact_begin=400, contact_end=720):
        """
        Splits dataframe into two sets, one for making churn predictions and one for creating prioritized contact list

        Parameters:
            dataframe: merged pandas DF from previous steps
            churn_end: allows user to select defined churn timeframe, default is 400 days from last visit
            contact_end: allows user to select end of timeframe for contact list, default is 720 days from last visit (2 years)

        Returns:
             Two pandas dataframes for use with model predictions and prioritized contact list
        """

        for_model = dataframe[dataframe['Recency'].between(churn_begin, churn_end)]
        for_model.reset_index(drop=True, inplace=True)
        contact_list = dataframe[dataframe['Recency'].between(contact_begin, contact_end)]

        return for_model, contact_list

    def contact_transform(self, df, tenure_term=50, total_term=50, frequency_term=10):
        """
        Creates specific df for use as a prioiritzed contact list for dental staff.  Score is calculated as follows:
        Final Score = Recency (days) - Tenure/50 (days) - Total/50 ($) - Frequency/10 (visits)
        The lower the score, the higher the priority on the contact list.

        Parameters:
            df: contact_list df from previous step in chain
            tenure_term: term to shrink tenure value, higher term gives less weight to this value, default = 50
            total_term: term to shrink total value, higher term gives less weight to this value, default = 50
            frequency_term: term to shrink frequency value, higher term gives less weight to this value, default = 10

        Returns:
            Pandas df of patients sorted in prioritzed order for recontact based on calculated score
        """
        df = df.loc[:, ['PatNum', 'FName', 'Recency', 'Tenure', 'Total', 'Frequency']]
        df['Score'] = df['Recency'] - df['Tenure']/tenure_term - df['Total']/total_term - df['Frequency']/frequency_term
        df['FName'] = df['FName'].str[0].str.upper() + df['FName'].str[1:].str.lower()
        df.rename(columns={'FName':'First Name'}, inplace=True)
        df['Total'] = round(df['Total']).apply(lambda x : "${:,}".format(x))
        df = df.sort_values('Score')
        return df
