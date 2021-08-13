#!/usr/bin/env python
# coding: utf-8

# In[566]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import datetime
import time
import seaborn as sns


# In[567]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[568]:


df2016 = pd.read_csv('data/rtc_2016.csv')
df2017 = pd.read_csv('data/rtc_2017.csv')
df2018 = pd.read_csv('data/rtc_2018.csv')
df2019 = pd.read_csv('data/rtc_2019.csv')
df2020 = pd.read_csv('data/rtc_2020.csv')


# #### Aggregate Data

# In[569]:


total_df = pd.concat([df2016,df2017,df2018,df2019,df2020], ignore_index=True)


# In[571]:


total_df.drop('Unnamed: 0', inplace=True, axis=1)


# In[572]:


total_df.replace(0, np.nan, inplace=True)


# In[573]:


# Keep only the rows with at least 2 non-NA values.
# total_df.dropna(thresh=3,inplace=True)
total_df.dropna(subset=["crash_time","report_time", "arrival_time","total_involved","causes"],inplace=True)


# #### Time Splitter and Cleaner Function

# In[574]:


new_crash_time = total_df['crash_time'].astype(str).apply(lambda x: re.sub("[^0-9]", "", x))


# In[575]:


new_arrival_time = total_df['arrival_time'].astype(str).apply(lambda x: re.sub("[^0-9]", "", x))


# In[576]:


new_response_time = total_df['response_time'].astype(str).apply(lambda x: re.sub("[^0-9]", "", x))


# In[577]:


total_df['crash_time'] = new_crash_time.apply(np.int64)


# In[578]:


total_df['arrival_time'] = new_arrival_time.apply(np.int64)


# In[579]:


total_df['report_time'] = total_df['report_time'].round(0).astype(int)


# In[580]:


total_df['date'] = total_df['date'].astype(str)


# In[581]:


def clean_date(date_given):
    try:
        return pd.to_datetime(date_given,errors="coerce")
    except:
        return '0'
    


# In[582]:


total_df['date'] = total_df['date'].apply(lambda x: clean_date(x))


# In[583]:


# Four(4 Data Points with missing date)
total_df[np.isnat(total_df['date'])]


# In[584]:


total_df['month'] = pd.to_numeric(total_df['date'].dt.month.astype(int, errors='ignore'))
total_df['day'] = pd.to_numeric(total_df['date'].dt.day.astype(int, errors='ignore'))
total_df['year'] = pd.to_numeric(total_df['date'].dt.year.astype(int, errors='ignore'))


# In[585]:


total_df['month'].fillna(method='bfill',inplace=True)
total_df['day'].fillna(method='bfill', inplace=True)
total_df['year'].fillna(method='bfill', inplace=True)


# In[586]:


total_df['year'] = total_df['year'].apply(np.int64)
total_df['month'] = total_df['month'].apply(np.int64)
total_df['day'] = total_df['day'].apply(np.int64)


# In[587]:


def fill_missing_date(date, year, month, day):
    if pd.isnull(date):
        new_date = datetime.datetime(year=year, month=month, day=day)
        return new_date
    else:
        return date


# In[588]:


total_df['date'] = total_df.apply(lambda x:fill_missing_date(x['date'], x['year'], x['month'], x['day']), axis=1)


# In[589]:


total_df['crash_time'].apply(lambda x: len(str(x))).unique()


# In[590]:


total_df['report_time'].apply(lambda x: len(str(x))).unique()


# In[591]:


total_df['arrival_time'].apply(lambda x: len(str(x))).unique()


# In[592]:


total_df['response_time'].apply(lambda x: len(str(x))).unique()


# In[593]:


total_df[total_df['crash_time'].astype(str).map(len)==5]


# In[594]:


total_df[total_df['report_time'].astype(str).map(len)==5]


# In[595]:


total_df.loc[total_df['crash_time'].astype(str).map(len)==5, 'crash_time'] = 1123


# In[596]:


total_df.loc[total_df['report_time']==7855, 'report_time'] = 1855


# In[597]:


total_df.loc[total_df['report_time'].astype(str).map(len)==5, 'report_time'] = 1213


# In[598]:


total_df


# In[599]:


def time_splitter(time):
    to_string = str(time)
    if len(to_string) == 2:
        return pd.to_datetime("00" + ":" + str(time), format= '%H:%M')
    elif len(to_string) == 3:
        return pd.to_datetime(to_string[0] +":" + to_string[1:], format= '%H:%M')
    elif len(to_string) == 4:
        return pd.to_datetime(to_string[0:2] + ":"+ to_string[2:], format= '%H:%M')
    else:
        return pd.to_datetime(str(time), format= '%H:%M')


# In[600]:


total_df['crash_time'] = total_df['crash_time'].apply(lambda x:time_splitter(x)).dt.time


# In[601]:


total_df['report_time'] = total_df['report_time'].apply(lambda x:time_splitter(x)).dt.time


# In[602]:


total_df['arrival_time'] = total_df['arrival_time'].apply(lambda x:time_splitter(x)).dt.time


# In[603]:


def combine_date_time(date, time):
    return datetime.datetime.combine(date, time)

def subtract_date_time(start, end):
    return end - start


# In[604]:


crash_datetime = total_df.apply(lambda x: combine_date_time(x['date'], x['crash_time']), axis=1)
report_datetime = total_df.apply(lambda x: combine_date_time(x['date'], x['report_time']), axis=1)
arrival_datetime = total_df.apply(lambda x: combine_date_time(x['date'], x['arrival_time']), axis=1)


# In[605]:


total_df['report_minus_crash'] =  subtract_date_time(crash_datetime, report_datetime).apply(lambda x:x.total_seconds()).astype(int)


# In[606]:


total_df['arrival_minus_crash'] =  subtract_date_time(crash_datetime, arrival_datetime).apply(lambda x:x.total_seconds()).astype(int)


# In[607]:


total_df["arrival_minus_report"] =  subtract_date_time(report_datetime, arrival_datetime).apply(lambda x:x.total_seconds()).astype(int)


# In[608]:


total_df['datetime'] = pd.to_datetime(crash_datetime)


# In[609]:


total_df.replace(np.nan, 0, inplace=True)


# In[610]:


total_df['fleet_operator'] = total_df['fleet_operator'].replace(0, np.nan)
total_df['name_of_driver'] = total_df['name_of_driver'].replace(0, np.nan)
total_df['dl_no'] = total_df['dl_no'].replace(0, np.nan)


# In[611]:


total_df['vehicle_type']


# In[612]:


def get_automobile_no(car_det):
    try:
        # print(car_det)
        # Split the various car category
        car_det = car_det.split('&')
        all_g= []
        automobile_no = 0
        # Iterate through each item
        for item in car_det:
            item = item.strip()
            # Separate inner lists
            item = re.split(',| * ',item)
            # Iterate through the inner loop and add to the primary list
            if type(item) is list:
                for val in item:
                    val = val.strip()
                    if val not in ["","*",'HIT','RUN', '(HIT', 'RUN)']:
                        all_g.append(val)
            # Iterate through ther list generate including the numbers
        for each in all_g: 
            if each.isdigit():
                each = int(each)
                automobile_no += each - 1
            else: 
                automobile_no +=1
        if automobile_no < 1:
            return 0
        else:      
            return automobile_no
    except:
        return 0



# In[613]:


total_df['no_automobile'] = total_df.apply(lambda x: get_automobile_no(x['vehicle_cat']), axis=1)


# In[614]:


total_df['route'] = total_df['route'].str.strip()


# In[615]:


total_df[total_df['vehicle_cat']== '3COM']


# In[616]:


total_df.to_csv('data/cleaned_aggregated.csv', index=False)


# In[617]:


c.columns


# In[618]:


total_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




