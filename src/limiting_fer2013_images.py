#  Limiting fer2013 dataset images to 2000 per emotion.

import csv
import pandas as pd

ds = pd.read_csv(r"../csv_files/fer2013.csv")


def trainortest(train_data):
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

    for data in range(len(train_data)):
        if train_data[data][0] == 0:
            train_data[data][0] = 0
        elif train_data[data][0] == 1 or train_data[data][0] == 2 or train_data[data][0] == 5:
            train_data[data][0] = 1
        elif train_data[data][0] == 4:
            train_data[data][0] = 2
        elif train_data[data][0] == 3:
            train_data[data][0] = 3
        elif train_data[data][0] == 6:
            train_data[data][0] = 4

        csvData = train_data[data]

        with open('../csv_files/fer2013LimitedImages.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(csvData)

        csvFile.close()


train = ds[["emotion", "pixels", "Usage"]].values[ds["Usage"] == "Training"]
test = ds[["emotion", "pixels", "Usage"]].values[ds["Usage"] == "PublicTest"]

csvData = ['emotion', 'pixels', 'Usage']
with open(r'../csv_files/fer2013LimitedImages.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(csvData)

csvFile.close()

trainortest(train)
trainortest(test)
