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
            
            