#!/usr/bin/env python
# coding: utf-8

# ### PART 1 
# Installing and Importing programs to read and parse html document

# In[1]:


import requests


# In[2]:


get_ipython().system('pip install BeautifulSoup4')
get_ipython().system('pip install lxml')


# In[3]:


website_url = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M").text
from bs4 import BeautifulSoup
soup = BeautifulSoup(website_url,'lxml')
#print(soup.prettify())


# #### Using tag 'td' to find and split values. Assign values to the empty lists (i,e., code_list)

# In[4]:


import pandas as pd
codes_list=[]
borough_list=[]
neighborhood_list=[]
i=1
for tag in soup.table.find_all('td'):
    if i == 1:
        tag = tag.text
        tag = tag.replace('\n', '')
        codes_list.append(tag)
        
    if i == 2:
        tag = tag.text
        tag = tag.replace('\n', '')
        borough_list.append(tag)
    if i == 3: 
        tag = tag.text
        tag = tag.replace('\n', '')
        neighborhood_list.append(tag)
    i = i+1
    if i==4:
        i=1
#combining lists into dataframe
df = pd.DataFrame()
df['PostalCode']=codes_list
df['Borough']=borough_list 
df['Neighborhood']=neighborhood_list
#replacing / with ,
df=df.replace({'/':','},regex=True)
df


# #### Dropping not assigned Borough and reseting index

# In[5]:


df = df[df['Borough'] != 'Not assigned'].reset_index(drop=True)
df.head(12)


# In[6]:


df.shape


# ### PART 2

# ##### installing geocoder to get coordinates

# In[7]:


get_ipython().system('conda install -c conda-forge geocoder --yes')


# In[8]:


import geocoder


# In[9]:


latitude=[]
longitude=[]
# select each row of df 
for i in range(len(df)):
#initialize lat_lng_coords to none
    lat_lng_coords = None    
#loop until you get the coordinates
    while(lat_lng_coords is None):
        g=geocoder.arcgis('{},Toronto, Ontario'.format(df['PostalCode'][i]))
        lat_lng_coords=g.latlng
    latitude.append(lat_lng_coords[0])
    longitude.append(lat_lng_coords[1])
df['Latitude']=latitude
df['Longitude']=longitude


# In[10]:


print(df.shape)
df.head()


# ### PART 3

# ##### choosing Borough that contains word 'York' only

# In[11]:


df_York=df[df['Borough'].str.contains('York')].reset_index(drop=True)
print(df_York.shape)
df_York.head(10)


# In[12]:


get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library


# ##### Map of all neighborhoods from Borough that contains word 'York' only

# In[13]:



# create map of Manhattan using latitude and longitude values
map_df_York=folium.Map(location=[43.7001114, -79.4162979], zoom_start=10)

# add markers to map
for lat, lng, label in zip(df_York['Latitude'], df_York['Longitude'], df_York['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_df_York)  
    
map_df_York


# ### get venues near the neighborhoods

# In[14]:


# @hidden_cell
CLIENT_ID =   # your Foursquare ID
CLIENT_SECRET =  # your Foursquare Secret
VERSION = '20180605' # Foursquare API version


# In[15]:


LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# #### Number of unique venues

# In[16]:


york_venues = getNearbyVenues(names=df_York['Neighborhood'],
                                   latitudes=df_York['Latitude'],
                                   longitudes=df_York['Longitude']
                                  )
print('\n There are {} uniques categories.'.format(len(york_venues['Venue Category'].unique())))


# #### Getting the top 10 venues for each neighborhood

# In[17]:


york_onehot = pd.get_dummies(york_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
york_onehot['Neighborhood'] = york_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [york_onehot.columns[-1]] + list(york_onehot.columns[:-1])
york_onehot = york_onehot[fixed_columns]

york_onehot.head()


# In[21]:


york_grouped = york_onehot.groupby('Neighborhood').mean().reset_index()
york_grouped


# In[22]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[23]:


import numpy as np
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = york_grouped['Neighborhood']

for ind in np.arange(york_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(york_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# #### Clustering Neighborhoods using 3 clusters

# In[24]:


from sklearn.cluster import KMeans 
# set number of clusters
kclusters = 3

york_grouped_clustering = york_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(york_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[25]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

york_merged = df_York

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
york_merged = york_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood',how='right')

york_merged.head(10) # check the last columns!


# ####  Visualize the resulting clusters

# In[26]:


import matplotlib.cm as cm
import matplotlib.colors as colors
# create map
map_clusters = folium.Map(location=[43.7001114, -79.4162979], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(york_merged['Latitude'], york_merged['Longitude'], york_merged['Neighborhood'], york_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# #### Cluster 0 red color. Neigborhoods like Parks

# In[27]:


york_merged.loc[york_merged['Cluster Labels'] == 0, york_merged.columns[[2] + list(range(5, york_merged.shape[1]))]]


# #### Cluster 1 purple color. Neigborhoods like Coffe Shops

# In[28]:


york_merged.loc[york_merged['Cluster Labels'] == 1, york_merged.columns[[2] + list(range(5, york_merged.shape[1]))]]


# #### Cluster 2 teal color. Neigborhoods like Construction & Landscaping

# In[29]:


york_merged.loc[york_merged['Cluster Labels'] == 2, york_merged.columns[[2] + list(range(5, york_merged.shape[1]))]]


# In[ ]:




