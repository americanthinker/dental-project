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

        return merged

    def contact_list(self, filepath):
        appt = pd.read_csv(filepath, usecols=['PatNum', 'AptStatus', 'AptDateTime'])

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
        num_visits = appt.groupby('PatNum', as_index=False)['AptDateTime'].count().sort_values('AptDateTime', ascending=False)
        return num_visits


if __name__ == '__main__':
    t = Transform()
    cl = t.contact_list('../data/raw/appt.csv')
    print(cl.head())