import pandas as pd

def rename_collumns(df, new_collumns):
    for i, col in enumerate(df.columns):
        df = df.rename(columns={col: new_collumns[i]})
    return df

def make_numeric(df, column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce', downcast='integer')
    return df

def remove_nan(df):
    return df.dropna()

def remove_numeric_values(df,column_name, high, low):
    

    df = df[df[column_name] > low]

    df = df[df[column_name] < high] 
    return df 