import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder



def rename_collumns(df, new_collumns):
    for i, col in enumerate(df.columns):
        df = df.rename(columns={col: new_collumns[i]})
    return df

def stress_cleaner(df):
    df["Stress_c"] = df["Stress"]
    
    for i in df["Stress_c"]:
        if i.isdigit():
            if float(i) > 100 :
                df["Stress_c"] = df["Stress_c"].replace(i,'100')
        else:
            df["Stress_c"] = df["Stress_c"].replace(i,'100')


    df["Stress_c"] = pd.to_numeric(df["Stress_c"], errors='coerce', downcast='integer')
        
    return df

def se_cleaner(df):
    df["Self esteem_c"] = df["Self esteem"]
    for i in df["Self esteem_c"]:
            if i.isdigit():
                if float(i) > 100 :
                    df["Self esteem_c"] = df["Self esteem_c"].replace(i,'100')
            elif "/" in i or "cent" in i:
                df["Self esteem_c"] = df["Self esteem_c"].replace(i,'0')

            elif len(i.split()) > 1:
                for j in i.split():
                    if j.isdigit():
                        df["Self esteem_c"] = df["Self esteem_c"].replace(i,j)
            elif "," in i:
                df["Self esteem_c"] = df["Self esteem_c"].replace(i,i[0])
            elif "€" in i:
                df["Self esteem_c"] = df["Self esteem_c"].replace(i,i[1])
            
            if "More" in i or "%" in i or "Win" in i or "Twice" in i or "All" in i or "BTC" in i or "1.000.000" in i:
                df["Self esteem_c"] = df["Self esteem_c"].replace(i,"100")

            if "equally" in i or "Depends" in i or "depends" in i or "less" in i or "not" in i or "None" in i or "-1" in i:
                df["Self esteem_c"] = df["Self esteem_c"].replace(i,"0")
            if "enough" in i:
                df["Self esteem_c"] = df["Self esteem_c"].replace(i,"100")
            
    df["Self esteem_c"] = pd.to_numeric(df["Self esteem_c"], errors='coerce', downcast='integer')

    return df

def RN_cleaner(df):
    df["RN_c"] = df["RN"]

    for i in df["RN_c"]:
        if i.isdigit():
            if float(i)>10 or float(i)<0:
                df["RN_c"] = df["RN_c"].replace(i,"NAN")
        else:
            df["RN_c"] = df["RN_c"].replace(i,"NAN")
    df["RN_c"] = pd.to_numeric(df["RN_c"], errors='coerce', downcast='integer')

    for i in df["RN_c"]:
        print(i)
    return df

def programme_cleaner(df):
    df["Programme_c"] = df["Programme"]

    for i in df["Programme_c"] :
        if "computational" in i.lower():
            df["Programme_c"] = df["Programme_c"].replace(i,"CLS")
        elif "artificial" in i.lower() or "ai" in i.lower():
            df["Programme_c"] = df["Programme_c"].replace(i,"AI")
        elif "econometrics" in i.lower():
            df["Programme_c"] = df["Programme_c"].replace(i,"Econometrics")
        elif "business analytics" in i.lower():
            df["Programme_c"] = df["Programme_c"].replace(i,"BA")
        elif "bioinformatics" in i.lower():
            df["Programme_c"] = df["Programme_c"].replace(i,"BI")
        elif "human" in i.lower():
            df["Programme_c"] = df["Programme_c"].replace(i,"HLT") 
        elif "risk management" in i.lower() or "qrm" in i.lower():
            df["Programme_c"] = df["Programme_c"].replace(i,"QRM") 
        elif "computer science" in i.lower() or "cs" in i.lower() :
            df["Programme_c"] = df["Programme_c"].replace(i,"CS") 
        elif "information" in i.lower() or "data" in i.lower():
            df["Programme_c"] = df["Programme_c"].replace(i,"DS")
        elif "fin" in i.lower() or "f&t" in i.lower():
            df["Programme_c"] = df["Programme_c"].replace(i,"FT")
        elif len(i) > 5:
            df["Programme_c"] = df["Programme_c"].replace(i,i[0:5])
        
    for i in df["Programme_c"].unique():

        if list(df["Programme_c"]).count(i) == 1:
            df["Programme_c"] = df["Programme_c"].replace(i,"other")
    
    for i in df["Programme_c"]:
        print(i)
    

    return df


            
            
            
          

def remove_nan(df):
    return df.dropna()

def categorical(df, col, course=True):
    # creating instance of one-hot-encoder
    enc = OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)
    enc_df = pd.DataFrame(enc.fit_transform(df[[col]]).toarray())# merge with main df bridge_df on key values
    new_df = df.join(enc_df)
    
    
    if course:
        new_df = new_df.rename(columns={0:f"{col},no"})
        new_df = new_df.rename(columns={1:f"{col},yes"})
        new_df = new_df.rename(columns={2:f"{col},uk"})

    else:
        for i in range(len(df["Programme_c"].unique())):
            new_df = new_df.rename(columns={i:df["Programme_c"].unique()[i]})

    print(new_df)
    return new_df
if __name__ == "__main__":
    pass