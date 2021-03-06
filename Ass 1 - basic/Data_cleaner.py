import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import dateparser as dp
import re



def classify_numericals(df, col):
    df[f"{col}_cat"] = df[f"{col}"]

    for i in df[f"{col}_cat"]:
        if float(i)>0 and float(i)<34:
            df[f"{col}_cat"] = df[f"{col}_cat"].replace(i,'low')
        elif float(i) > 33 and float(i) < 67:
            df[f"{col}_cat"] = df[f"{col}_cat"].replace(i,'med')
        else:
            df[f"{col}_cat"] = df[f"{col}_cat"].replace(i,'high')
    
    return df



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

    #for i in df["RN_c"]:
        #print(i)
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
    


    return df


def neighbors_cleaner(df):
    df['Neighbors_c'] = df['Neighbours'].fillna(0)
    df.iloc[32, df.columns.get_loc('Neighbors_c')] = 0
    df.iloc[79, df.columns.get_loc('Neighbors_c')] = 0  
    df.iloc[218, df.columns.get_loc('Neighbors_c')] = 1
    df.iloc[219, df.columns.get_loc('Neighbors_c')] = 0
    df.iloc[298, df.columns.get_loc('Neighbors_c')] = 2
    df.iloc[266, df.columns.get_loc('Neighbors_c')] = 0
    
    df = df.replace({'Neighbors_c' : { 8979937 : 0, 300 : 0, 265 : 0, 200: 0}})
    
    
    df["Neighbors_c"] = pd.to_numeric(df["Neighbors_c"])

def birth_date_cleaner(df):
    #column with all full birthdays
    df['Birthday_c'] = np.nan
    #columns with all birhtdays , months and days
    df['Birthdate_dm_c'] = np.nan
    
    for i in range(len(df)): 
        date = cleanup_bday(df['Birthday'].iloc[i])
        if date is not None and len(date) == 3:
            df['Birthday_c'].iloc[i] = "{0}-{1}-{2}".format(date[0], date[1], date[2])
            df['Birthdate_dm_c'].iloc[i] = "{0}-{1}".format(date[0], date[1])
        elif date is not None and len(date) ==2:
            df['Birthdate_dm_c'].iloc[i] = "{0}-{1}".format(date[0], date[1])
   

def cleanup_bday(date):
    #handling some exceptions that are readable by humans
    if date == 20051989:
            return ([20, 5, 1989])
    elif date ==23011999:
            return [23, 1, 1999]
    elif date ==19951124:
            return [24, 11, 1995]
    try: 
        #returning exceptions that are no date
        return str(int(date))
    except ValueError: 
        pass
    
    
    date_2 = dp.parse(date, settings={'STRICT_PARSING': True})
    if date_2 is not None:
        return([date_2.day, date_2.month, date_2.year])
        #return "{0}-{1}-{2}".format(date_2.day, date_2.month, date_2.year)
    else:
        date_3 = dp.parse(date,settings={'REQUIRE_PARTS': ['day', 'month']})
        if date_3 is not None:
            return([date_3.day, date_3.month])
        
    if date =='November 23rd, nineteen hundred eighty nine':
        return [23, 11, 1989]

    
    return None

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

    return new_df

def bedtime_parser(df):
    Hours = []
    for time in df["Bedtime"]:
        time_int  = re.findall(r'[0-9]+', time)
        if len(time_int):
            hour_int = int(time_int[0])
            if hour_int == 10:
                hour_int = 22
            if hour_int == 11:
                hour_int = 23
            if hour_int == 12:
                hour_int = 24
            if hour_int > 24:
                Hours.append(23)
            else:
                Hours.append(hour_int)
        else:
            Hours.append(23)
    
    
            
    df["Bedtime_Hour_c"] = Hours
    return df

def calc_age(df):
    year = df['Birthday_c'].str.split('-', expand=True)[2]
    year = year.fillna(2021)
    df['Age'] =  (2021 - year.astype(int))
    df['Age'][df["Age"] > 60] = 0
    
    return df


def binarize(df):
    df = df.replace({'Stat' : { 'mu' : 2, 'sigma' : 0, 'unknown' : 1}})
    df["Stat"] = pd.to_numeric(df["Stat"])
    #df = df.replace({'ML' : { 'yes' : 1, 'no' : 0}})
    df = df.replace({'DB' : { 'ja' : 2, 'nee' : 0, 'unknown' : 1}})
    df["DB"] = pd.to_numeric(df["DB"])
    df = df.replace({'ML' : { 'yes' : 2, 'no' : 0, 'unknown' : 1}})
    df["ML"] = pd.to_numeric(df["ML"])
    df = df.replace({'Stand up' : { 'yes' : 2, 'no' : 0, 'unknown' : 1}})
    df["Stand up"] = pd.to_numeric(df["Stand up"])
    df = df.replace({'Gender' : { 'female' : 2, 'male' : 0, 'unknown' : 1}})
    df["Gender"] = pd.to_numeric(df["Gender"])
    
    return df

def time(df):
    df_temp = pd.to_datetime('3/30/2021 23:59:59') - pd.to_datetime(df['Time'])
    df['Time_c'] = df_temp.dt.total_seconds()
    
    df['Time_c'] = df['Time_c']/86400
    
    return(df)
    
if __name__ == "__main__":
    pass