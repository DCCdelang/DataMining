import pandas as pd

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

    return df


            
            
            
          

def remove_nan(df):
    return df.dropna()

def remove_numeric_values(df,column_name, high, low):

    df = df[df[column_name] > low]

    df = df[df[column_name] < high] 
    return df 
if __name__ == "__main__":
    pass