#!/usr/bin/env python
# coding: utf-8

# # Task 1: Оценка качества воздуха в г. Павлодар
# Цель: Провести EDA и анализ исторических данных о загрязнении.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('ggplot')
# matplotlib inline


# In[3]:


data_dir = 'data'
pollutants = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'h2s', 'o3']
df_list = []

# Читаем уже распакованные .csv файлы
for p in pollutants:
    file_path = os.path.join(data_dir, f"{p}.csv")
    if os.path.exists(file_path):
        temp_df = pd.read_csv(file_path)
        # Фильтруем Павлодар
        temp_df = temp_df[temp_df['city'].str.contains('Pavlodar', case=False, na=False)]
        if not temp_df.empty:
            temp_df['pollutant'] = p
            df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
print(f"Загружено {len(df)} записей для Павлодара.")


# In[4]:


# Сводная таблица по дням
df_pivot = df.pivot_table(index='datetime_utc', columns='pollutant', values='value_ugm3', aggfunc='mean')
df_daily = df_pivot.resample('D').mean()
df_daily.head()


# In[ ]:


# Визуализация PM2.5 и нормы ВОЗ
plt.figure(figsize=(15, 6))
if 'pm2_5' in df_daily.columns:
    sns.lineplot(data=df_daily, x=df_daily.index, y='pm2_5', label='PM 2.5')
plt.axhline(y=15, color='r', linestyle='--', label='WHO PM2.5 Guideline (15 µg/m³)')
plt.title('Концентрация PM2.5 в Павлодаре')
plt.ylabel('µg/m³')
plt.legend()
plt.show()


# In[ ]:


# Матрица корреляций
plt.figure(figsize=(8, 6))
sns.heatmap(df_daily.corr(), annot=True, cmap='coolwarm')
plt.title('Корреляция между поллютантами')
plt.show()

