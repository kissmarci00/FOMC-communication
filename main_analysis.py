#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 19:17:51 2025

@author: kissmarcell
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 18:56:55 2025

@author: kissmarcell
"""
import pandas as pd
from datetime import datetime
import statsmodels.api as sm
from sklearn.linear_model import Lasso


#Import classified texts
text_ml=pd.read_excel("text_classified_joined.xlsx")

text_ml['date']=pd.to_datetime(text_ml['date_text'].astype(str), format='%Y%m%d')
text_ml['date'] = text_ml['date'].dt.date
text=text_ml

#Set according to analyzed event
conf=1

if conf==0:
    
    text=text[text['type']=="statement"]
    
if conf==1: 
    text=text[text['type']!="statement"]

# Step 1: Filter and count hawkish and dovish answers for topic 'mp'
mp_data = text[text['predicted_label_topic'] == 'policy']
mp_counts = mp_data.groupby('date')['predicted_label_mp'].value_counts().unstack(fill_value=0)
mp_counts_unc=mp_data.groupby('date')[['Uncertainy_Word_Count','Total_Word_Count']].sum()
mp_counts_unc['unc_mp']=mp_counts_unc['Uncertainy_Word_Count']/mp_counts_unc['Total_Word_Count']

# Step 2: Filter and count positive and negative answers for topic 'ec'
ec_data = text[text['predicted_label_topic'] == 'econ']
ec_counts = ec_data.groupby('date')['predicted_label_econ'].value_counts().unstack(fill_value=0)
ec_counts_unc=ec_data.groupby('date')[['Uncertainy_Word_Count','Total_Word_Count']].sum()
ec_counts_unc['unc_ec']=ec_counts_unc['Uncertainy_Word_Count']/ec_counts_unc['Total_Word_Count']

mp_counts.columns = ["dovish", "hawkish","neutral_mp"]
ec_counts.columns = ["negative","neutral_ec","positive"]


mp_counts_reset = mp_counts.reset_index()
ec_counts_reset = ec_counts.reset_index()
mp_counts_unc_reset = mp_counts_unc.reset_index()
ec_counts_unc_reset = ec_counts_unc.reset_index()


# Merge on the 'date' column, keeping all rows from mp_counts and ec_counts
text_ml= pd.merge(mp_counts_reset, ec_counts_reset, on='date', how='left').fillna(0)
text_ml= pd.merge(text_ml, ec_counts_unc_reset[['date','unc_ec']], on='date', how='left').fillna(0)
text_ml= pd.merge(text_ml, mp_counts_unc_reset[['date','unc_mp']], on='date', how='left').fillna(0)


text_ml['sent_mp']=text_ml['dovish']-text_ml['hawkish']
text_ml['sent_ec']=text_ml['positive']-text_ml['negative']
text_ml['all_mp']=(text_ml['dovish']+text_ml['hawkish'])
text_ml['all_ec']=(text_ml['positive']+text_ml['negative'])
text_ml['score_ml_mp']=(text_ml['sent_mp']/text_ml['all_mp']).fillna(0)
text_ml['score_ml_ec']=(text_ml['sent_ec']/text_ml['all_ec']).fillna(0)

text_ml=text_ml[['date','score_ml_mp','score_ml_ec','unc_mp','unc_ec']]    # Step 2: Filter and count positive and negative answers for topic 'ec'
    
    
    
#Import asset price changes and constructed shocks

if conf==0:
    my_shocks=pd.read_excel("Data/cieslak_ann.xlsx")
    my_shocks['date']=pd.to_datetime(my_shocks['date']).dt.date

    my_changes=pd.read_excel("Data/ann_changes.xlsx")
    my_changes['date'] = pd.to_datetime(my_changes['date'].astype(str), format='%Y%m%d').dt.date
else:
    my_shocks=pd.read_excel("Data/cieslak_conf.xlsx")
    my_shocks['date']=pd.to_datetime(my_shocks['date']).dt.date

    my_changes=pd.read_excel("Data/conf_changes.xlsx")
    my_changes['date'] = pd.to_datetime(my_changes['date'].astype(str), format='%Y%m%d').dt.date


#Import controls


controls=pd.read_excel("Data/controls_fomc.xlsx")
controls['date']=controls['date'].dt.date

#Import shocks from previous papers for robustness checks

outside_shocks=pd.read_excel("Data/shocks_fomc.xlsx")
outside_shocks['date']=outside_shocks['date'].dt.date

df=pd.merge(my_shocks,text_ml,on='date',how='left').fillna(0)
df=pd.merge(df,my_changes,on='date',how='left')
df=pd.merge(df,controls,on='date',how='left')
df=pd.merge(df,outside_shocks,on='date',how='left')


#Import Voice tone Score 

vts=pd.read_excel("Data/voicetonescores.xlsx")
vts['date']=vts['date'].dt.date
df=pd.merge(df,vts,on='date',how='left')


#Construct sentiment shocks

# Define independent variables


X_columns = [
    "dUNEMP1", "dPCE1", "dg1", "dcpi1_d1", "dunemp1_d1",
    "dcpi5_d1", "dg1_d2", "prev_mid",
    "prev_uemid", "prev_uedis", "effr", "dfinstress", "dbci","dvix","depu","sentdiff_news"
]

# Run regression for score_ml_mp
df["score_ml_mp_lag"] = df["score_ml_mp"].shift(1).fillna(0)  # Lagged dependent variable
df["score_ml_ec_lag"] = df["score_ml_ec"].shift(1).fillna(0)  # Lagged dependent variable

X_columns_mp = X_columns + ["score_ml_mp_lag"]  + ["score_ml_ec_lag"] # Include lagged term

y_mp = df["score_ml_mp"]  # Dependent variable
X_mp = df[X_columns_mp]  # Independent variables

X_mp = sm.add_constant(X_mp)  # Add intercept
model_mp = Lasso(alpha=0.00)  # alpha is the regularization strength
model_mp.fit(X_mp, y_mp)

# Residuals
news_mp_shock = y_mp - model_mp.predict(X_mp)

# Print results manually
print("\n=== Lasso Coefficients ===")
for col, coef in zip(X_mp.columns, model_mp.coef_):
    print(f"{col}: {coef}")


# Run regression for score_ml_ec
df["score_ml_mp_lag"] = df["score_ml_mp"].shift(1).fillna(0) 
df["score_ml_ec_lag"] = df["score_ml_ec"].shift(1).fillna(0)  # Lagged dependent variable
X_columns_ec = X_columns + ["score_ml_ec_lag"] +  ["score_ml_mp_lag"]  # Include lagged term

y_ec = df["score_ml_ec"]  # Dependent variable
X_ec = df[X_columns_ec]  # Independent variables

X_ec = sm.add_constant(X_ec)  # Add intercept


# Fit Lasso regression
model_ec = Lasso(alpha=0.00)  # alpha is the regularization strength
model_ec.fit(X_ec, y_ec)

# Residuals
news_ec_shock = y_ec - model_ec.predict(X_ec)

# Print results manually
print("\n=== Lasso Coefficients ===")
for col, coef in zip(X_ec.columns, model_ec.coef_):
    print(f"{col}: {coef}")

# Run regression for unc_ec
df["unc_ec_lag"] = df["unc_ec"].shift(1).fillna(0) 
df["unc_mp_lag"] = df["unc_mp"].shift(1).fillna(0)  # Lagged dependent variable
X_columns_ec = X_columns + ["unc_mp_lag"]  # Include lagged term

y_ec = df["unc_ec"]  # Dependent variable
X_ec = df[X_columns_ec]  # Independent variables

X_ec = sm.add_constant(X_ec)  # Add intercept
model_ec = Lasso(alpha=0.00)  # alpha is the regularization strength
model_ec.fit(X_ec, y_ec)

# Residuals
news_ec_unc_shock = y_ec - model_ec.predict(X_ec)

# Print results manually
print("\n=== Lasso Coefficients ===")
for col, coef in zip(X_ec.columns, model_ec.coef_):
    print(f"{col}: {coef}")


# Run regression for unc_mp
df["unc_ec_lag"] = df["unc_ec"].shift(1).fillna(0) 
df["unc_mp_lag"] = df["unc_mp"].shift(1).fillna(0)  # Lagged dependent variable
X_columns_ec = X_columns + ["unc_mp_lag"]  # Include lagged term

y_ec = df["unc_mp"]  # Dependent variable
X_ec = df[X_columns_ec]  # Independent variables

X_ec = sm.add_constant(X_ec)  # Add intercept
model_ec = Lasso(alpha=0.00)  # alpha is the regularization strength
model_ec.fit(X_ec, y_ec)

# Residuals
news_mp_unc_shock = y_ec - model_ec.predict(X_ec)

# Print results manually
print("\n=== Lasso Coefficients ===")
for col, coef in zip(X_ec.columns, model_ec.coef_):
    print(f"{col}: {coef}")
if conf==1:
    # Run regression for unc_mp
    df["avg_lag"] = df["avg"].shift(1).fillna(0)  # Lagged dependent variable
    X_columns_ec = X_columns + ["avg_lag"]  # Include lagged term
    
    y_ec = df["avg"]  # Dependent variable
    X_ec = df[X_columns_ec]  # Independent variables
    
    X_ec = sm.add_constant(X_ec)  # Add intercept
    model_ec = Lasso(alpha=0.00)  # alpha is the regularization strength
    model_ec.fit(X_ec, y_ec)
    
    # Residuals
    avg_shock = y_ec - model_ec.predict(X_ec)

df['chair'] = df['date_text'].apply(
    lambda d: 'Bernanke' if d < 20140201 else (
        'Powell' if d > 20180201 else 'Yellen'
    )
)




chair_dummies = pd.get_dummies(df['chair'], prefix='chair', drop_first=True).astype(int)


# Join the dummies to the original DataFrame
df = pd.concat([df, chair_dummies], axis=1)


#Define controls for final regression


X_columns = ["avg_shock","news_mp_shock","news_ec_shock","news_mp_unc_shock","news_ec_unc_shock","ddis","deffr","epu_level","duemid","dmid"]




# Define dependent variable
Y_column = "mp"



# Ensure the DataFrame has these columns
df["news_mp_shock"] = news_mp_shock # Add residuals from the previous regression
df["news_ec_shock"] = news_ec_shock 
df["news_mp_unc_shock"] = news_mp_unc_shock  # Add residuals from the previous regression
df["news_ec_unc_shock"] = news_ec_unc_shock 
df["avg_shock"]=avg_shock



df_temp = df.dropna(subset=[Y_column] + X_columns)


# Define X and y
y = df_temp[Y_column]  # Dependent variable
X = df_temp[X_columns]  # Independent variables

# Add a constant for the intercept
X=sm.add_constant(X)

# Fit the OLS regression model
model = sm.OLS(y, X).fit(cov_type='HC1')

# Print regression results
print(model.summary())








