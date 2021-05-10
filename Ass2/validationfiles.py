import pandas as pd

def make_files(df,x):
    if x == 'test':
        df = df.tail(80000)
    else:
        df = df.head(20000)
    return df

def add_values(df, name):
    values = []
    for i in range(len(df['click_bool'])):
        value = 0
        if df.iloc[i]['click_bool'] != 0:
            value +=1
        if df.iloc[i]['booking_bool'] != 0:
            value +=4
        values.append(value)

    df['value'] = values
    df.to_csv(name, index=False)

def drop_nan_columns():
    df = pd.read_csv('Data/training_set_VU_DM.csv')
    # plot(df)
    print(df.shape)
    df1 = df.dropna(axis=1, thresh= 0.1 * df.shape[0])
    print(df1.shape)
    for i in list(df.columns):
        if i not in (df1.columns):
            print(i)
    df1.to_csv('Data/training_set_VU_DM_deleted.csv', index=False)

def make_clicked_file():
    df = pd.read_csv('Data/processed_train.csv')
    df = df.loc[df['click_bool'] == 1]
    df.to_csv('Data/clicked_data.csv', index=False)

def drop_columns(df):
    kut_columns = ['click_bool', 'gross_bookings_usd', 'booking_bool', 'count', 'booking_bool_tot', 'count_tot','srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff','date_time','Unnamed: 0']
    
    df = df.drop(kut_columns, axis=1)
