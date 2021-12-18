# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 20:31:32 2021

HHA_507_ANOVA_Assigmnet_4

@author: Alejandro Herrera

"""
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import kurtosis
from scipy.stats import skew, bartlett
import statsmodels.stats.multicomp as mc
import statsmodels.api as sm
## DATASET on Youth Smoking in US
df = pd.read_csv('C:/Users/Alejandro Herrera/Documents/HHA_507_ STATISTICS_FOR_HEALTH_INFORMATICS/Youth_Tobacco_Survey__YTS__Data.csv')
list(df)
['YEAR',
 'LocationAbbr',
 'LocationDesc',
 'TopicType',
 'TopicDesc',
 'MeasureDesc',
 'DataSource',
 'Response',
 'Data_Value_Unit',
 'Data_Value_Type',
 'Data_Value',
 'Data_Value_Footnote_Symbol',
 'Data_Value_Footnote',
 'Data_Value_Std_Err',
 'Low_Confidence_Limit',
 'High_Confidence_Limit',
 'Sample_Size',
 'Gender',
 'Race',
 'Age',
 'Education',
 'GeoLocation',
 'TopicTypeId',
 'TopicId',
 'MeasureId',
 'StratificationID1',
 'StratificationID2',
 'StratificationID3',
 'StratificationID4',
 'SubMeasureID',
 'DisplayOrder']

sample_size = df.sample(100)
list(['Data_Value'])
#1 factor of Eduction has 2 levels
df.Education.value_counts()
'''Middle School    5633
   High School      4967
   Name: Education, dtype: int64'''
#Status has 4 levels
df.MeasureDesc.value_counts()
'''Smoking Status    4107
   User Status       4083
   Percent of Current Smokers Who Want to Quit    1284
   Quit Attempt in Past Year Among Current Cigarette Smokers    1126'''
#Response has 3 levels
df.Response.value_counts()
'''Ever        2730
   Frequent    2730
   Current     2730'''
df.Data_Value.value_counts()

#Save as dataframe:
#Location Abbriviation has 50 levels
df.LocationAbbr.value_counts()
len(df.LocationAbbr.value_counts())
locationscounts = pd.DataFrame(df.LocationAbbr.value_counts())
locationscounts = locationscounts.reset_index()

data = pd.DataFrame(df.Data_Value.value_counts())

#Transroming data column from float to int
df['Data_Value'] = pd.to_numeric(df['Data_Value'])
## 1. Find a new data set that contains at least: 
#Data frame with 10,600 rows and 31 variables
''' VARIABLES FOR 1 WAY ANOVA TESTS 
#Dependent variable 1 (continuous value) = 'Data_Value'
#Independent variable 1 (categorical value) = 'Response'
#Indepdendent variable 2 (categorical value) = 'Education
#Indepdendent variable 3 (categorical value) ='MeasureDesc'

'''
## 3. Conduct at least 3 1-way anova’s. For each ANOVA:
#=============================================================================
#ANOVA 1: Is there a difference the levels between Education and Percent of 
#youth smoking?
#=============================================================================

####CHECKING ASSUMPTIONS####
model = smf.ols("Data_Value ~ C(Education)", data=df).fit()
stats.shapiro(model.resid)
# ShapiroResult(statistic=0.869617223739624, pvalue=0.0)

model = smf.ols("Data_Value ~ C(Education)", data = df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
'''
                    sum_sq       df           F         PR(>F)
C(Education)  2.581435e+05      1.0  610.953532  5.247471e-131
Residual      4.258213e+06  10078.0         NaN            NaN
'''
# The difference between the Education and Data_Value is significant due to the P value being less thn 0.05 allowing us to reject the null hypothesis.

#Graphical Representation of noramlitly
#Different levels of education:
edu1 = df[df['Education'] == 'Middle School']
edu2 = df[df['Education'] == 'High School']


plt.hist(edu1['Data_Value'])
plt.show()

plt.hist(edu2['Data_Value'])
plt.show()

###Kurtosis
kurtosis(edu1['Data_Value'])
kurtosis(edu2['Data_Value'])

print(kurtosis(edu1['Data_Value']))
print(kurtosis(edu2['Data_Value']))

print(skew(edu1['Data_Value']))
print(skew(edu2['Data_Value']))

##Bartlett test
stats.bartlett(edu1['Data_Value'],
               edu2['Data_Value'])
# BartlettResult(statistic=69.14252777036864, pvalue=9.159874781624161e-17)
#Template
stats.f_oneway(edu1['Data_Value'],
               edu2['Data_Value'])

#Post-hoc analysis for significant differences between groups
# TUKEY HONESTLY SIGNIFICANT DIFFERENCE (HSD)

comp = mc.MultiComparison(df['Data_Value'], df['Education'])
post_hoc_res = comp.tukeyhsd()
tukey1way = pd.DataFrame(post_hoc_res.summary())

edu1['Data_Value'].describe()
edu2['Data_Value'].describe()
#==============================================================================
#ANOVA 2: Is there a difference the levels between MeasureDesc and Percent of 
#youth smoking?
#=============================================================================
####CHECKING ASSUMPTIONS####
model_1 = smf.ols("Data_Value ~ C(MeasureDesc)", data=df).fit()
stats.shapiro(model_1.resid)

#ShapiroResult(statistic=0.8820940256118774, pvalue=0.0)
model_1 = smf.ols("Data_Value ~ C(MeasureDesc)", data = df).fit()
anova_table = sm.stats.anova_lm(model_1,typ=2)
anova_table
'''
                    sum_sq       df            F  PR(>F)
C(MeasureDesc)  2.831451e+06      3.0  5644.171382     0.0
Residual        1.684906e+06  10076.0          NaN     NaN
'''
# The difference between the MeasureDesc and Data_Value is significant due to the P value being less thn 0.05 allowing us to reject the null hypothesis.
#Graphical Representation of noramlitly
#Different levels of MeasureDesc:
StatusSmoke1 = df[df['MeasureDesc'] == 'Smoking Status']
StatusSmoke2 = df[df['MeasureDesc'] == 'User Status']
StatusSmoke3 = df[df['MeasureDesc'] == 'Percent of Current Smokers Who Want to Quit']
StatusSmoke4 = df[df['MeasureDesc'] == 'Quit Attempt in Past Year Among Current Cigarette Smokers']

plt.hist(StatusSmoke1['Data_Value'])
plt.show()

plt.hist(StatusSmoke2['Data_Value'])
plt.show()

plt.hist(StatusSmoke3['Data_Value'])
plt.show()

plt.hist(StatusSmoke4['Data_Value'])
plt.show()

###Kurtosis
kurtosis(StatusSmoke1['Data_Value'])
kurtosis(StatusSmoke2['Data_Value'])
kurtosis(StatusSmoke3['Data_Value'])
kurtosis(StatusSmoke4['Data_Value'])

print(kurtosis(StatusSmoke1['Data_Value']))
print(kurtosis(StatusSmoke2['Data_Value']))
print(kurtosis(StatusSmoke3['Data_Value']))
print(kurtosis(StatusSmoke4['Data_Value']))
#Skewness
print(skew(StatusSmoke1['Data_Value']))
print(skew(StatusSmoke2['Data_Value']))
print(skew(StatusSmoke3['Data_Value']))
print(skew(StatusSmoke4['Data_Value']))
##Bartlett test
stats.bartlett(StatusSmoke1['Data_Value'],
               StatusSmoke2['Data_Value'],
               StatusSmoke3['Data_Value'],
               StatusSmoke4['Data_Value'])
# BartlettResult(statistic=2650.508663616527, pvalue=0.0)
#Template
stats.f_oneway(StatusSmoke1['Data_Value'],
               StatusSmoke2['Data_Value'],
               StatusSmoke3['Data_Value'],
               StatusSmoke4['Data_Value'])
#Post-hoc analysis for significant differences between groups
# TUKEY HONESTLY SIGNIFICANT DIFFERENCE (HSD)

comp = mc.MultiComparison(df['Data_Value'], df['MeasureDesc'])
post_hoc_res = comp.tukeyhsd()
tukey1way = pd.DataFrame(post_hoc_res.summary())

StatusSmoke1['Data_Value'].describe()
StatusSmoke2['Data_Value'].describe()
StatusSmoke3['Data_Value'].describe()
#=======================================================
#ANOVA 3: Is there a difference the levels between Location by Response and Percent(Data_Value) of 
#youth smoking?
#=======================================================
####CHECKING ASSUMPTIONS####
model_2 = smf.ols("Data_Value ~ C(Response)", data=df).fit()
stats.shapiro(model_2.resid)
#ShapiroResult(statistic=0.910588264465332, pvalue=0.0)
model_2 = smf.ols("Data_Value ~ C(Response)", data = df).fit()
anova_table = sm.stats.anova_lm(model_2,typ=2)
anova_table 
'''
                   sum_sq      df           F  PR(>F)
C(Response)  7.060446e+05     2.0  2641.14584     0.0
Residual     1.093894e+06  8184.0         NaN     NaN'''
#The difference between the Response and Data_Value is significant due to the P value being less thn 0.05 allowing us to reject the null hypothesis.
#Graphical Representation of noramlitly
#Different levels of e:
response1 = df[df['Response'] == 'Ever']
response2 = df[df['Response'] == 'Frequent']
response3 = df[df['Response'] == 'Current']

plt.hist(response1['Data_Value'])
plt.show()

plt.hist(response2['Data_Value'])
plt.show()

plt.hist(response3['Data_Value'])
plt.show()

###Kurtosis
kurtosis(response1['Data_Value'])
kurtosis(response2['Data_Value'])
kurtosis(response3['Data_Value'])

print(kurtosis(response1['Data_Value']))
print(kurtosis(response2['Data_Value']))
#3.2709303537209227
print(kurtosis(response3['Data_Value']))
#0.704060475933753
#Skewness
print(skew(response1['Data_Value']))
print(skew(response2['Data_Value']))
#1.8366331153203845
print(skew(response3['Data_Value']))
#1.1603012785964415
##Bartlett test
stats.bartlett(response1['Data_Value'],
               response2['Data_Value'],
               response3['Data_Value'])
#BartlettResult(statistic=5387.142915622402, pvalue=0.0)
#Template
stats.f_oneway(response1['Data_Value'],
               response2['Data_Value'],
               response3['Data_Value'])
#Post-hoc analysis for significant differences between groups
# TUKEY HONESTLY SIGNIFICANT DIFFERENCE (HSD)

comp = mc.MultiComparison(df['Data_Value'], df['Response'])
post_hoc_res = comp.tukeyhsd()
tukey1way = pd.DataFrame(post_hoc_res.summary())

response1['Data_Value'].describe()
response2['Data_Value'].describe()
response3['Data_Value'].describe()