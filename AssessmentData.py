import pandas as pd
import re 
import geopandas as gpd
import pipeline_jmidkiff as pipeline


pd.set_option('display.max_rows', 100)

assessments = pd.read_csv(
    'data/Cook_County_Assessor_s_Residential_Property_Characteristics.zip', 
    sep='\t', compression='zip')
pipeline.show(assessments)

acs = pd.read_csv('data/ACS Data.csv')


col_list = ['Property Class', 'Wall Material', 'Roof Material', 
        'Repair Condition', 'Renovation', 'Prior Tax Year Market Value Estimate (Land)', 
        'Prior Tax Year Market Value Estimate (Building)', 'Land Square Feet', 
        'Building Square Feet', 'Age']
for col in col_list: 
    print(pipeline.describe(assessments[col]))
    print("-" * 30)

pipeline.group_count(assessments, 'Age').sort_values(ascending=False)
# Important features: 
# Percentages of those in block group: distinct Property Class, 
# distinct wall material, distinct roof material, Repair Condition, Renovation

# Counts: 
# Total units in block group. 

# Medians & Means: 
# Prior Year Market Value Estimate (Land), & Building, Land Square Feet, 
# Building Square Feet, 

