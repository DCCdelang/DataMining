import pandas as pd
import csv


with open('Ass 1 - basic/Data/SmsCollection.csv', newline='') as f:
    label = []
    messages = []
    reader = csv.reader(f)
    for row in reader:
        for i in row:
            
            message = ''
            s = i.split(";")
            for j in s:
                if j == 'ham' or j == 'spam':
                    label.append(j)
                else:
                    message = message + j
        messages.append(message)

print(label[0], messages[0])
messages.pop(0)
df = pd.DataFrame()
df["label"] = label
df["text"] = messages
df.to_csv("Sms_new.csv")