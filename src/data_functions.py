import pandas as pd
import matplotlib.pyplot as plt

class wrangle:
    def __init__(self):
        pass
    
    def downcast(self, df):
        float_cols = df.dtypes[df.dtypes == float].index
        int_cols = df.dtypes[df.dtypes == int].index
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
    def todate(self, df, list_of_cols):
        for col in list_of_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    def patient_dropper(self, df, target_col, targets):
        for target in targets:
            df = df[df[target_col] != target]
        return df
            
    def scaler(column, bottom_range, top_range):
        '''
        Scales data between a range between (bottom_range, top_range)
        𝑥𝑛𝑜𝑟𝑚𝑎𝑙𝑖𝑧𝑒𝑑=(𝑏−𝑎) * 𝑥−𝑚𝑖𝑛(𝑥)     + a
                        𝑚𝑎𝑥(𝑥)−𝑚𝑖𝑛(𝑥)
        Input: pd.Series, np.array (list will not broadcast)
        Ouput: scaled version of Input between bottom_range and top_range
        '''
        #This value is known due to the current (Nov 2020) SOF distribution
        # ***** Need to not hard code this value *****
        max_series = column.max()
        min_series = column.min()
        multiplier = top_range - bottom_range
        numerator = series - min_series
        denominator = max_series - min_series
        ans = (multiplier * numerator/denominator) + bottom_range
        return ans + bottom_range