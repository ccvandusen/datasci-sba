
# coding: utf-8

# Playing around with the SBA dataset
# CDC: https://www.sba.gov/category/lender-navigation/steps-sba-lending/cdc504-loans (Certified Development Company)
# 7a vs 504 loan: http://www.hcdc.com/sba-504-vs-sba-7a-loan-programs/

# 

# In[1]:

get_ipython().magic('matplotlib inline')

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pylab as plt
import csv
from pylab import *


# In[2]:

df = pd.read_excel('./data/FOIA - 504 (FY1991-Present).xlsx', converters={'BorrZip':str})


# In[3]:

df.head()


# In[4]:

df = df.loc[df['CDC_State'] == 'CA']


# In[5]:

df.head()


# In[ ]:




# In[9]:

df.BusinessType.unique()


# In[10]:

df.LoanStatus.unique()


# In[11]:

df['JobsSupported'].groupby(df['LoanStatus']).describe()


# In[12]:

df['paid_in_full'] = (df['LoanStatus'] == 'PIF')
df['charged_off'] = (df['LoanStatus'] == 'CHGOFF')


# In[13]:

formula = 'JobsSupported ~ paid_in_full'
results = smf.ols(formula, data=df).fit()
results.summary()


# In[14]:

df['InitialInterestRate'].groupby(df['LoanStatus']).describe()


# In[15]:

formula = 'JobsSupported ~ InitialInterestRate'
results = smf.ols(formula, data=df).fit()
results.summary()


# In[16]:

plt.scatter(x=df.InitialInterestRate, y=df.JobsSupported)


# In[35]:

x = df['charged_off'].groupby(df['BorrZip']).count()


# In[36]:

x


# In[37]:

test_attr = {k: x.loc[k] for k in x.index}
#print(d)


# In[38]:

def read_ascii_boundary(filestem):
    '''Reads polygon data from an ASCII boundary file.
    
    Returns a dictionary with polygon IDs for keys. The value for each
    key is another dictionary with three keys:
    'name' - the name of the polygon
    'polygon' - list of (longitude, latitude) pairs defining the main
    polygon boundary
    'exclusions' - list of lists of (lon, lat) pairs for any exclusions in
    the main polygon
    '''
    metadata_file = filestem + 'a.dat'
    data_file = filestem + '.dat'
    # Read metadata
    lines = [line.strip().strip('"') for line in open(metadata_file)]
    polygon_ids = lines[::6]
    polygon_names = lines[2::6]
    polygon_data = {}
    for polygon_id, polygon_name in zip(polygon_ids, polygon_names):
        # Initialize entry with name of polygon.
        # In this case the polygon_name will be the 5-digit ZIP code.
        polygon_data[polygon_id] = {'name': polygon_name}
    del polygon_data['0']
    # Read lon and lat.
    f = open(data_file)
    for line in f:
        fields = line.split()
        if len(fields) == 3:
            # Initialize new polygon
            polygon_id = fields[0]
            polygon_data[polygon_id]['polygon'] = []
            polygon_data[polygon_id]['exclusions'] = []
        elif len(fields) == 1:
            # -99999 denotes the start of a new sub-polygon
            if fields[0] == '-99999':
                polygon_data[polygon_id]['exclusions'].append([])
        else:
            # Add lon/lat pair to main polygon or exclusion
            lon = float(fields[0])
            lat = float(fields[1])
            if polygon_data[polygon_id]['exclusions']:
                polygon_data[polygon_id]['exclusions'][-1].append((lon, lat))
            else:
                polygon_data[polygon_id]['polygon'].append((lon, lat))
    return polygon_data


# In[51]:

# Example of making a map
# Read in ZIP code boundaries for California
d = read_ascii_boundary('./data/zip5/zt06_d00')

# Read in data for number of births by ZIP code in California
max_test_attr = max(test_attr.values())

# Create figure and two axes: one to hold the map and one to hold
# the colorbar
figure(figsize=(8, 8), dpi=1000)
map_axis = axes([0.0, 0.0, 0.8, 0.9])
cb_axis = axes([0.83, 0.1, 0.03, 0.6])

# Define colormap to color the ZIP codes.
# You can try changing this to cm.Blues or any other colormap
# to get a different effect
cmap = cm.PuRd

# Create the map axis
axes(map_axis)
axis([-125, -114, 32, 42.5])
gca().set_axis_off()

# Loop over the ZIP codes in the boundary file
for polygon_id in d:
    polygon_data = array(d[polygon_id]['polygon'])
    zipcode = d[polygon_id]['name']
    num_test_attr = test_attr[zipcode] if zipcode in test_attr else 0.
    # Define the color for the ZIP code
    fc = cmap(num_test_attr / max_test_attr)
    # Draw the ZIP code
    patch = Polygon(array(polygon_data), facecolor=fc,
        edgecolor=(.3, .3, .3, 1), linewidth=.2)
    gca().add_patch(patch)
title('Test Attr per ZIP Code in California (2007)')

# Draw colorbar
cb = mpl.colorbar.ColorbarBase(cb_axis, cmap=cmap,
    norm = mpl.colors.Normalize(vmin=0, vmax=max_test_attr))
cb.set_label('Test Attr')

# Change all fonts to Arial
for o in gcf().findobj(matplotlib.text.Text):
    o.set_fontname('Arial')

# Export figure to bitmap
#savefig('../images/ca_births.png')


# In[ ]:




# In[ ]:




# In[58]:

# Read in data for number of births by ZIP code in California
f = csv.reader(open('./data/CA_2007_births_by_ZIP.txt'))
births = {}
# Skip header line
header = next(f)
# Add data for each ZIP code
for row in f:
    zipcode, totalbirths = row
    births[zipcode] = float(totalbirths)
max_births = max(births.values())


# In[59]:

births


# In[ ]:



