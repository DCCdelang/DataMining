import pandas as pd
import seaborn as sns

df = pd.read_csv('Ass2/Data/training_set_VU_DM.csv')
df2 = pd.read_csv('Ass2/Data/clicked_data.csv')
df3 = df
important = list(df2['srch_id'])



for i in range(1, 332785 + 1):
    if i not in important:
        df3 = df[df.srch_id != i]
    else:
        important.remove(i)
    if i %1000 == 0:
        print(i)
    
df3.to_csv('cool.csv')

