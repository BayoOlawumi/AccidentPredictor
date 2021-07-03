#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df2016 = pd.read_excel('data/RTC.xlsx', sheet_name='2016')
df2017 = pd.read_excel('data/RTC.xlsx', sheet_name='2017')
df2018 = pd.read_excel('data/RTC.xlsx', sheet_name='2018')
df2019 = pd.read_excel('data/RTC.xlsx', sheet_name='2019')
df2020 = pd.read_excel('data/RTC.xlsx', sheet_name='2020')


# ### Data Cleaning

# ### **** 2016 ****

# In[236]:


df2016.drop(index= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], axis=0, inplace=True) 


# In[237]:


columns = ['sn','command', 'date','crash_time','report_time','arrival_time','response_time','route','location','vehicle_no','vehicle_type','vehicle_cat','vehicle_make','vehicle_model','fleet_operator','name_of_driver', 'dl_no','causes', 'no_injured_male_adult','no_injured_female_adult','no_injured_male_child','no_injured_female_child', 'total_injured', 'no_killed_male_adult','no_killed_female_adult','no_killed_male_child','no_killed_female_child', 'total_killed', 'no_involved_male_adult','no_involved_female_adult','no_involved_male_child','no_involved_female_child', 'total_involved']


# In[238]:


df2016.columns = columns


# In[239]:


# Fill the name of command for all rows
def fill_command(df_c):
    global current_command
    if df_c != np.empty:
        if str(df_c).lower().startswith('rs'):
            current_command = df_c
        return current_command
    else:
        return


# In[240]:


df2016.command = df2016.apply(lambda x: fill_command(x['command']), axis=1)


# In[241]:


df2016.sn = pd.to_numeric(df2016.sn, errors="coerce")


# In[242]:


df2016 = df2016[df2016['sn'].notnull()]


# In[243]:


df2016.sn = df2016.sn.astype(int)


# In[273]:


df2016.to_csv("data/rtc_2016.csv")


# ### **** 2017 ****

# In[245]:


df2017.drop(index= [0,1,2,3,4,5,6,7], axis=0,inplace=True) 


# In[246]:


df2017.columns = columns


# In[247]:


df2017.command = df2017.apply(lambda x: fill_command(x['command']), axis=1)


# In[248]:


df2017.sn = pd.to_numeric(df2017.sn, errors="coerce")


# In[249]:


df2017 = df2017[df2017['sn'].notnull()]


# In[250]:


df2017.sn = df2017.sn.astype(int)


# In[274]:


df2017.to_csv("data/rtc_2017.csv")


# ### **** 2018 ****

# In[251]:


df2018.drop(index= [0,1,2,3,4,5,6,7], axis=0,inplace=True) 


# In[252]:


df2018.columns = columns


# In[253]:


df2018.command = df2018.apply(lambda x: fill_command(x['command']), axis=1)


# In[254]:


df2018.sn = pd.to_numeric(df2018.sn, errors="coerce")


# In[255]:


df2018 = df2018[df2018['sn'].notnull()]


# In[256]:


df2018.sn = df2018.sn.astype(int)


# In[270]:


df2018.to_csv("data/rtc_2018.csv")


# ### **** 2019 ****

# In[258]:


df2019.drop(index= [0,1,2,3,4,5,6], axis=0,inplace=True) 


# In[259]:


df2019.columns = columns


# In[260]:


df2019.command = df2019.apply(lambda x: fill_command(x['command']), axis=1)


# In[261]:


df2019.sn = pd.to_numeric(df2019.sn, errors="coerce")


# In[262]:


df2019 = df2019[df2019['sn'].notnull()]


# In[263]:


df2019.sn = df2019.sn.astype(int)


# In[271]:


df2019.to_csv("data/rtc_2019.csv")


# ### **** 2020 ****

# In[264]:


df2020.drop(index= [0,1,2,3,4,5,6,7], axis=0,inplace=True) 


# In[265]:


df2020.columns = columns


# In[266]:


df2020.command = df2020.apply(lambda x: fill_command(x['command']), axis=1)


# In[267]:


df2020.sn = pd.to_numeric(df2020.sn, errors="coerce")


# In[268]:


df2020 = df2020[df2020['sn'].notnull()]


# In[269]:


df2020.sn = df2020.sn.astype(int)


# In[272]:


df2020.to_csv("data/rtc_2020.csv")


# In[ ]:




