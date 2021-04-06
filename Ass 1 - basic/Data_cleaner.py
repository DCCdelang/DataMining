import pandas as pd
class Data_cleaner:
    def __init__(self,dataframe):
        self.dataframe = dataframe
   
    def rename_collumns(self, new_collumns):
        for i, col in enumerate(self.dataframe.columns):
            self.dataframe = self.dataframe.rename(columns={col: new_collumns[i]})
        return self.dataframe
    
    def make_numeric(self, column_name):
        self.dataframe[column_name] = pd.to_numeric(self.dataframe[column_name], errors='coerce', downcast='integer')
        return self.dataframe

    def remove_nan(self):
        return self.dataframe.dropna()

    def remove_numeric_values(self,column_name, high, low):
        

        self.dataframe = self.dataframe[self.dataframe[column_name] > low]

        self.dataframe = self.dataframe[self.dataframe[column_name] < high] 
        return self.dataframe 