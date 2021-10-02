from csv_ical import Convert
import time
import pandas as pd

convert = Convert()
convert.CSV_FILE_LOCATION = "vinay.csv"
convert.SAVE_LOCATION = "F21_schedule.ics"
convert.read_ical(convert.SAVE_LOCATION)
convert.make_csv()
convert.save_csv(convert.CSV_FILE_LOCATION)

df = pd.read_csv("vinay.csv")
df.columns = ["Event","StartT ime","End Time","Description","Location"]
