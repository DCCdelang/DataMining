import pandas as pd

def make_files(df, kind, test=500000, train=30000):
    if kind == 'test':
        df = df.tail(test)
        df = add_values(df)
    if kind == "dante":
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
    # df = pd.read_csv('Data/prepro_train.csv')
    clicked = df.loc[df['click_bool'] == 1]
    
    non_clicked = df.loc[df['click_bool'] == 0]
    non_clicked = non_clicked.sample(3*len(clicked['srch_id']))

    fifty_fifty = pd.concat([clicked, non_clicked])
    fifty_fifty = fifty_fifty.sort_values(by=['srch_id'])
    return fifty_fifty
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
    df = pd.read_csv('Data/prepro_train.csv')

    df1 = df.tail(500000)

    df1 = make_files(df1, 'dante')
    df1.to_csv('Data/validation_test_25_75.csv', index=False)
    
    # df = pd.read_csv('Data/clicked_data.csv')

    # df = make_files(df, 'test')
    # df.to_csv('Data/validation_train_clicked.csv', index=False)

    # print('2')
    df2 = df.iloc[:-500000]

    print(df.tail())
    print(df1.tail())
    print(df2.tail())

    fifty_fifty = make_50_50_file(df2)
    fifty_fifty.to_csv('Data/25_75_small.csv', index=False)

    # fifty_fifty = pd.read_csv('Data/25_75.csv')
    fifty_fifty2 = make_50_50_file(df)
    fifty_fifty2.to_csv('Data/25_75.csv', index=False)

    print(df.shape,df1.shape,df2.shape,fifty_fifty2.shape)

    # fifty_fifty.to_csv('Data/25_75_small.csv', index=False)
    # makes submission_train set
    # df = pd.read_csv('Data/prepro_train2.csv')
    # df = add_values(df)
    # df.to_csv('Data/train_submission.csv', index=False)
    # print('3')

    lijst_val = df1.srch_id.unique()
    lijst_fifty = fifty_fifty.head(200000).srch_id.unique()

    print(len(lijst_val), len(lijst_fifty))
    print(set(lijst_val) & set(lijst_fifty))

