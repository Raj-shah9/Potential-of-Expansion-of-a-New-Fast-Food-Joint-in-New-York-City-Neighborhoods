# -*- coding: utf-8 -*-
"""BAX 422 452 Final Project



# WebScrapping Part
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import requests
#from bs4 import BeautifulSoup
import os
#!conda install -c conda-forge folium=0.5.0 --yes
import folium 
#!conda install -c conda-forge geopy --yes
from geopy.geocoders import Nominatim 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
# %matplotlib inline
import seaborn as sns


from sklearn.cluster import KMeans



print('Done!')

"""#### Let's first scrape the **zipcode** data of New York City and save them in a pandas dataframe."""

def get_new_york_data():
    url='https://cocl.us/new_york_dataset'
    resp=requests.get(url).json()
    # all data is present in features label
    features=resp['features']
    # define the dataframe columns
    column_names = ['Borough', 'Neighborhood', 'Latitude', 'Longitude'] 
    # instantiate the dataframe
    new_york_data = pd.DataFrame(columns=column_names)
    for data in features:
        borough = data['properties']['borough'] 
        neighborhood_name = data['properties']['name']
        neighborhood_latlon = data['geometry']['coordinates']
        neighborhood_lat = neighborhood_latlon[1]
        neighborhood_lon = neighborhood_latlon[0]
        new_york_data = new_york_data.append({'Borough': borough,
                                          'Neighborhood': neighborhood_name,
                                          'Latitude': neighborhood_lat,
                                          'Longitude': neighborhood_lon}, ignore_index=True)
    return new_york_data

ny_data = get_new_york_data()
ny_data

ny_data.to_csv('ny_datatocsv.csv')

"""#### Using **uszipcode** package to convert zipcode information into coordinates for mapping."""

CLIENT_ID = '0LDLJMSIXSX1WFN1AOOM1RIKJJUO2F0YTHRAL4NYLJIZ221T' 
CLIENT_SECRET = 'PN4AYLZ5GE3QWKRUWODQAAUDIWDDCK0GEQIXLTFMIIFK3QU3a' 
CODE = 'JYSZJDXY0T43XP4EAYKHVGJA54BI0ZTXV3TKET5ERZVCWQ0Y#_=_'
ACCESS_TOKEN = 'IPLYRE5DMQH0LS3M5MRUS0GGJN45ZJ2DLRRDMNPE1TNSFCVT' 
VERSION = '20180604'
LIMIT = 100
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)

#API Key:fsq3d8hyd0YCTTWKZHzz0CKfC7Q2Qc7VizV6pbR9d6PZ/pg=

def geo_location(address):
    # get geo location of address
    geolocator = Nominatim(user_agent="foursquare_agent")
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    return latitude,longitude


def get_venues(lat,lng):
    #set variables
    radius=400
    LIMIT=100
    #url to fetch data from foursquare api
    url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&oauth_token={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
    # get all the data
    results = requests.get(url).json()
    venue_data=results["response"]['groups'][0]['items']
    venue_details=[]
    for row in venue_data:
        try:
            venue_id=row['venue']['id']
            venue_name=row['venue']['name']
            venue_category=row['venue']['categories'][0]['name']
            venue_details.append([venue_id,venue_name,venue_category])
        except KeyError:
            pass
    column_names=['ID','Name','Category']
    df = pd.DataFrame(venue_details,columns=column_names)
    return df

"""http://foursquare-categories.herokuapp.com 

Burger places is listed under "Burger Joint" and 'Fast Food Restaurants'

Don't run this cell multiple times than needed!!!  LIMITED CALL NUMBERS FOR FOURSQUARE ACCOUNT
"""

# column_names=['Borough', 'Neighborhood', 'ID','Name']
# burger_joint_ny=pd.DataFrame(columns=column_names)
# count=1
# for row in ny_data.values.tolist():
#     Borough, Neighborhood, Latitude, Longitude=row
#     venues = get_venues(Latitude,Longitude)
#     burger_joints=venues[venues['Category']=='Burger Joint']   
#     print('(',count,'/',len(ny_data),')','Fast Food Restaurants in '+Neighborhood+', '+Borough+':'+str(len(burger_joints)))
#     print(row)
#     for resturant_detail in Fast_Food_Restaurants.values.tolist():
#         id, name , category=resturant_detail
#         burger_joint_ny = burger_joint_ny.append({'Borough': Borough,
#                                                 'Neighborhood': Neighborhood, 
#                                                 'ID': id,
#                                                 'Name' : name
#                                                }, ignore_index=True)
#     count+=1

# column_names=['Borough', 'Neighborhood', 'ID','Name']
# Fast_Food_Rest_ny=pd.DataFrame(columns=column_names)
# count=1
# for row in ny_data.values.tolist():
#     Borough, Neighborhood, Latitude, Longitude=row
#     venues = get_venues(Latitude,Longitude)
#     Fast_Food_Restaurants=venues[venues['Category']=='Fast Food Restaurant']   
#     print('(',count,'/',len(ny_data),')','Fast Food Restaurants in '+Neighborhood+', '+Borough+':'+str(len(Fast_Food_Restaurants)))
#     print(row)
#     for resturant_detail in Fast_Food_Restaurants.values.tolist():
#         id, name , category=resturant_detail
#         Fast_Food_Rest_ny = Fast_Food_Rest_ny.append({'Borough': Borough,
#                                                 'Neighborhood': Neighborhood, 
#                                                 'ID': id,
#                                                 'Name' : name
#                                                }, ignore_index=True)
#     count+=1

"""Save them to CSV and do not access the API except necessary"""

# burger_joint_ny.to_csv('burger_joint_ny_tocsv1.csv')

# Fast_Food_Rest_ny.to_csv('Fast_Food_Rest_ny_tocsv1.csv')

burger_joint_ny = pd.read_csv('burger_joint_ny_tocsv1.csv', index_col=0)
Fast_Food_Rest_ny = pd.read_csv('Fast_Food_Rest_ny_tocsv1.csv', index_col=0)

competitors_df = pd.concat([burger_joint_ny, Fast_Food_Rest_ny],ignore_index = True)
competitors_df

def get_venue_details(venue_id):
    #url to fetch data from foursquare api
    url = 'https://api.foursquare.com/v2/venues/{}?&client_id={}&client_secret={}&oauth_token={}&v={}'.format(
            venue_id,
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION)
    # get all the data
    results = requests.get(url).json()
    print(results)
    venue_data=results['response']['venue']
    venue_details=[]
    try:
        venue_id=venue_data['id']
        venue_name=venue_data['name']
        venue_likes=venue_data['likes']['count']
        venue_rating=venue_data['rating']
        venue_tips=venue_data['tips']['count']
        venue_lat = venue_data['location']['lat']
        venue_long = venue_data['location']['lng']
        venue_details.append([venue_id,venue_name,venue_likes,venue_rating,venue_tips,venue_lat,venue_long])
    except KeyError:
        pass
    column_names=['ID','Name','Likes','Rating','Tips','Latitude','Longitude']
    df = pd.DataFrame(venue_details,columns=column_names)
    return df

def get_venue_details(venue_id):
    #url to fetch data from foursquare api
    url = 'https://api.foursquare.com/v2/venues/{}?&client_id={}&client_secret={}&oauth_token={}&v={}'.format(
            venue_id,
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION)
    # get all the data
    results = requests.get(url).json()
    print(results)
    venue_data=results['response']['venue']
    venue_details=[]
    try:
        venue_id=venue_data['id']
        venue_name=venue_data['name']
        venue_likes=venue_data['likes']['count']
        venue_rating=venue_data['rating']
        venue_tips=venue_data['tips']['count']
        venue_details.append([venue_id,venue_name,venue_likes,venue_rating,venue_tips])
    except KeyError:
        pass
    column_names=['ID','Name','Likes','Rating','Tips']
    df = pd.DataFrame(venue_details,columns=column_names)
    return df

# https://api.foursquare.com/v2/venues/55568a71498e3524400a356e?&client_id=0LDLJMSIXSX1WFN1AOOM1RIKJJUO2F0YTHRAL4NYLJIZ221T&client_secret=PN4AYLZ5GE3QWKRUWODQAAUDIWDDCK0GEQIXLTFMIIFK3QU3a&oauth_token=IPLYRE5DMQH0LS3M5MRUS0GGJN45ZJ2DLRRDMNPE1TNSFCVT&v=20180604
# CLIENT_ID = '0LDLJMSIXSX1WFN1AOOM1RIKJJUO2F0YTHRAL4NYLJIZ221T' 
# CLIENT_SECRET = 'PN4AYLZ5GE3QWKRUWODQAAUDIWDDCK0GEQIXLTFMIIFK3QU3a' 
# #ACCESS_TOKEN = 'IPLYRE5DMQH0LS3M5MRUS0GGJN45ZJ2DLRRDMNPE1TNSFCVT' 
# VERSION = '20180604'

url = 'https://api.foursquare.com/v2/venues/55568a71498e3524400a356e?&client_id=0LDLJMSIXSX1WFN1AOOM1RIKJJUO2F0YTHRAL4NYLJIZ221T&client_secret=PN4AYLZ5GE3QWKRUWODQAAUDIWDDCK0GEQIXLTFMIIFK3QU3a&oauth_token=IPLYRE5DMQH0LS3M5MRUS0GGJN45ZJ2DLRRDMNPE1TNSFCVT&v=20180604'
results = requests.get(url).json()

print(results)

column_names=['Borough', 'Neighborhood', 'ID','Name','Likes','Rating','Tips']
competitors_stats_df=pd.DataFrame(columns=column_names)

count=1

for row in competitors_df.values.tolist():
    Borough,Neighborhood,ID,Name=row
    try:
        venue_details=get_venue_details(ID)
        print(venue_details)
        id,name,likes,rating,tips=venue_details.values.tolist()[0]
    except IndexError:
        print('No data available for id=',ID)
        # we will assign 0 value for these resturants as they may have been 
        #recently opened or details does not exist in FourSquare Database
        id,name,likes,rating,tips=[0]*5
    print('(',count,'/',len(competitors_df),')','processed')
    competitors_stats_df = competitors_stats_df.append({'Borough': Borough,
                                                'Neighborhood': Neighborhood, 
                                                'ID': id,
                                                'Name' : name,
                                                'Likes' : likes,
                                                'Rating' : rating,
                                                'Tips' : tips
                                               }, ignore_index=True)
    count+=1
competitors_stats_df.tail()

column_names=['Borough', 'Neighborhood', 'ID','Name','Likes','Rating','Tips','Latitude','Longitude']

competitors_stats_df=pd.DataFrame(columns=column_names)

count=1
for row in competitors_df.values.tolist():
  Borough,Neighborhood,ID,Name=row
  try:
    venue_details=get_venue_details(ID)
    print(venue_details)
    id,name,likes,rating,tips,lat,long=venue_details.values.tolist()[0]
  except:
    print('No data available for id=',ID)
        # we will assign 0 value for these resturants as they may have been 
        #recently opened or details does not exist in FourSquare Database
    id,name,likes,rating,tips,lat,long =[0]*7
  print('(',count,'/',len(competitors_df),')','processed')
  competitors_stats_df = competitors_stats_df.append({'Borough': Borough,
                                                'Neighborhood': Neighborhood, 
                                                'ID': id,
                                                'Name' : name,
                                                'Likes' : likes,
                                                'Rating' : rating,
                                                'Tips' : tips,
                                                'Latitude':lat,
                                                'Longitude':long
                                               }, ignore_index=True)
  count+=1

competitors_stats_df.tail()

competitors_stats_df.to_csv('competitors_stats_dftocsv.csv')

competitors_stats_df = pd.read_csv('competitors_stats_dftocsv.csv')

competitors_stats_df

map = folium.Map(location=geo_location('New York'), zoom_start=11)
for lat, lng, neighborhood in zip(ny_data['Latitude'], ny_data['Longitude'], ny_data['Neighborhood']):
    label = '{}'.format(neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map) 

map

"""## Map all neighborhoods"""

map = folium.Map(location=geo_location('New York'), zoom_start=11)
for lat, lng, neighborhood in zip(ny_data['Latitude'], ny_data['Longitude'], ny_data['Neighborhood']):
    label = '{}'.format(neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map) 

map

"""### Objective 1. Cluster NYC Neighborhoods

#### Now let's explore San Francisco neighborhoods using the Foursquare API

#### Firstly, we define functions to get the Venues nearby each neighborhoods.
"""

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

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
            v['venue']['categories'][0]['name'],
            v['venue']['categories'][0]['icon']['prefix']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category',
                  'Venue Parent Category Link']
    
    return(nearby_venues)

ny_venues = getNearbyVenues(names=ny_data['Neighborhood'],
                                   latitudes=ny_data['Latitude'],
                                   longitudes=ny_data['Longitude']
                                  )

"""#### Take a glance of the venues around each neighborhoods. And count the total number of unique venue types within 500 meters of each NYC neighborhoods."""

print(ny_venues.shape)
ny_venues.head()

import re
ny_venues['Venue Parent Category'] = ny_venues['Venue Parent Category Link'].map(lambda x: re.sub(r'(https://ss3.4sqi.net/img/categories_v2/)(\w+)/(\w+)', r'\2', x))

ny_venues

ny_venues.groupby('Neighborhood').count()

print('There are {} uniques venue categories in NYC.'.format(len(ny_venues['Venue Category'].unique())))
print('There are {} uniques Parent venue categories in NYC.'.format(len(ny_venues['Venue Parent Category'].unique())))

"""#### Analyze each neighborhoods by turning venues types into dummy variables."""

# one hot encoding
ny_onehot = pd.get_dummies(ny_venues[['Venue Parent Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
ny_onehot['Neighborhood'] = ny_venues['Neighborhood'] 

# move neighborhood column to the first column
first_column = ny_onehot.pop('Neighborhood')
ny_onehot.insert(0, 'Neighborhood', first_column)

ny_onehot.head()

ny_onehot.shape

ny_grouped = ny_onehot.groupby('Neighborhood').mean().reset_index()
ny_grouped

"""#### First, let's write a function to sort the parent venues in descending order."""

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

"""Now let's create the new dataframe and display the venues frequency in descending for each neighborhood."""

num_par_venues = 9

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_par_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = ny_grouped['Neighborhood']

for ind in np.arange(ny_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(ny_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()

"""#### Now, let's cluster the neighborhoods.
#### To find out the optimum value for k, we use the elbow methods. We should select the value of k, after which the distortion/inertia start decreasing in a linear fashion.
"""

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

ny_grouped_clustering = ny_grouped.drop('Neighborhood', 1)

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}

K = range(1, 10)
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k, random_state=123).fit(ny_grouped_clustering)
    kmeanModel.fit(ny_grouped_clustering)
 
    distortions.append(sum(np.min(cdist(ny_grouped_clustering, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / ny_grouped_clustering.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(ny_grouped_clustering, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / ny_grouped_clustering.shape[0]
    mapping2[k] = kmeanModel.inertia_

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

from kneed import KneeLocator
kl = KneeLocator(
    K, distortions, curve="convex", direction="decreasing"
   )
print(kl.elbow)

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

from kneed import KneeLocator
kl = KneeLocator(
    K, inertias, curve="convex", direction="decreasing"
   )
print(kl.elbow)

"""### From the above graphs, especially using inertia method, we find **optimum number of clusters to be 4**

"""

# set number of clusters
kclusters = 4

ny_grouped_clustering = ny_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(ny_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]

try:
    neighborhoods_venues_sorted.drop('Cluster Labels', axis = 1,inplace = True)
except KeyError:
    pass
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

ny_merged = ny_data

# merge sf_grouped with sf_data to add latitude/longitude for each neighborhood
ny_merged = ny_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

ny_merged.head() # check the last columns!

centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns = cat_list)
centroid_df

# def return_most_common_venues(row, num_top_venues):
#     row_categories = row.iloc[1:]
#     row_categories_sorted = row_categories.sort_values(ascending=False)
    
#     return row_categories_sorted.index.values[0:num_top_venues]
# columns = ['Lable']
# for ind in np.arange(num_par_venues):
#     try:
#         columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
#     except:
#         columns.append('{}th Most Common Venue'.format(ind+1))

# centroid_df_venue_sorted = pd.DataFrame(columns=columns)
# centroid_df_venue_sorted['Lable'] = np.arange(centroid_df.shape[0])

# for ind in np.arange(centroid_df.shape[0]):
#   centroid_df_venue_sorted.iloc[ind, 1:] = return_most_common_venues(centroid_df.iloc[ind, :], 8)

# centroid_df_venue_sorted

"""Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

## Results and Discussion <a name="results"></a>

#### Finally, let's visualize the resulting clusters
"""

# create map
map_clusters = folium.Map(location=geo_location('New York'), zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
for lat, lon, neighborhood, cluster in zip(ny_merged['Latitude'], ny_merged['Longitude'], ny_merged['Neighborhood'], ny_merged['Cluster Labels']):
    label = folium.Popup(str(neighborhood) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters

"""Now, we can examine each cluster and determine the discriminating venue categories that distinguish each cluster. Based on the defining categories, we can then assign a name to each cluster.

#### Based on the above observation, we can discover that San Francisco is a small but vibrant city. 
* Neighborhoods within San Francisco are mostly homogenous, but each contain multicultural characteristics by themselves. This is indicated by the volumn and variety of diners, bars etc.

* Venues within each neighborhoods imply that people living in San Francisco lead a both vibrant and busy lifestyle, with coffee shops being the top 3 most common venues in almost every neighborhoods.

* People in San Francisco cares about physiques and health, since sports bars, Gyms and parks all around the city.
"""

