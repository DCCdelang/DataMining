import pandas as pd

def make_files(df, kind, test=120000, train=300000):
    if kind == 'test':
        df = df.tail(test)
    else:
        df = df.head(train)
    return df

def add_values(df):
    values = []
    for i in range(len(df['click_bool'])):
        value = 0
        if df.iloc[i]['click_bool'] != 0:
            value +=1
        if df.iloc[i]['booking_bool'] != 0:
            value +=4
        values.append(value)

    df['value'] = values
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
    df = pd.read_csv('Data/prepro_train.csv')
    df = df.loc[df['click_bool'] == 1]
    df.to_csv('Data/clicked_data.csv', index=False)

def drop_columns(df):
    kut_columns = ['date_time','Unnamed: 0']
    
    df = df.drop(kut_columns, axis=1)
    return df


if __name__ == "__main__":

    # Deletes all the stupid features of the pre processed files
    df = pd.read_csv('prepro_test.csv')
    df = drop_columns(df)
    df.to_csv('prepro_test.csv', index=False)

    df = pd.read_csv('prepro_train.csv')
    df = drop_columns(df)
    df.to_csv('prepro_train.csv', index=False)

    # Makes file with all clicked values for training
    make_clicked_file()

    # Makes validation test and train set
    df = pd.read_csv('Data/prepro_train.csv')

    df = make_files(df, 'test')
    df.to_csv('Data/validation_test.csv', index=False)
    
    df = pd.read_csv('Data/clicked_data.csv')

    df = make_files(df, 'train')
    df.to_csv('Data/validation_train.csv', index=False)

    # adds values to the validation sets (5 for booking, 1 for clicking)
    df = pd.read_csv('Data/validation_test.csv')
    df = add_values(df)
    df.to_csv('Data/validation_test.csv', index=False)

    df = pd.read_csv('Data/validation_train.csv')
    df = add_values(df)
    df.to_csv('Data/validation_train.csv', index=False)
