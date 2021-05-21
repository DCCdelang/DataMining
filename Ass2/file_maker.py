import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def make_files(df, kind, test=500000, train=30000):
    if kind == 'test':
        df = df.tail(test)
        df = add_values(df)
    else:
        df = df.head(train)
        df = add_values(df)
    return df

def add_values(df):
    
    df['value'] = df.apply(lambda row: row.click_bool + (row.booking_bool * 4), axis=1)
    return df

def drop_nan_columns(df):
    # plot(df)
    print(df.shape)
    df1 = df.dropna(axis=1, thresh= 0.1 * df.shape[0])
    print(df1.shape)
    for i in list(df.columns):
        if i not in (df1.columns):
            print(i)
    return df

def make_clicked_file():
    df = pd.read_csv('Data/prepro_train2.csv')
    df = df.loc[df['click_bool'] == 1]
    df.to_csv('Data/clicked_data.csv', index=False)

def make_50_50_file(df):
    clicked = df.loc[df['click_bool'] == 1]
    
    non_clicked = df.loc[df['click_bool'] == 0]
    non_clicked = non_clicked.sample(3*len(clicked['srch_id']))

    fifty_fifty = pd.concat([clicked, non_clicked])
    fifty_fifty = fifty_fifty.sort_values(by=['srch_id'])
    fifty_fifty.to_csv('Data/25_75.csv', index=False)
# def drop_columns(df):
#     kut_columns = ['date_time']
    
#     df = df.drop(kut_columns, axis=1)
#     return df


if __name__ == "__main__":
    
    # # Deletes all the stupid features of the pre processed files
    # df = pd.read_csv('Data/prepro_test.csv')
    # print("-2")
    # # df = drop_columns(df)
    # df.to_csv('Data/prepro_test.csv', index=False)

    # df = pd.read_csv('Data/prepro_train.csv')
    # # df = drop_columns(df)
    # df.to_csv('Data/prepro_train.csv', index=False)

    # print("-1")
    # # Makes file with all clicked values for training
    # make_clicked_file()

    # print('1')
    # # Makes validation test and train set
    df = pd.read_csv('Data/prepro_train3.csv')

    x = df.values #returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df_v = make_files(df, 'test')
    df_v.to_csv('Data/validation_test3_scaled.csv', index=False)
    
    # df = pd.read_csv('Data/clicked_data.csv')

    # df = make_files(df, 'test')
    # df.to_csv('Data/validation_train_clicked.csv', index=False)

    # print('2')

    make_50_50_file()

    fifty_fifty = pd.read_csv('Data/25_75.csv')
    fifty_fifty = fifty_fifty.sort_values('srch_id')
    fifty_fifty.head(400000).to_csv('Data/25_75_small.csv', index=False)
    # makes submission_train set
    # df = pd.read_csv('Data/prepro_train2.csv')
    # df = add_values(df)
    # df.to_csv('Data/train_submission.csv', index=False)
    # print('3')


