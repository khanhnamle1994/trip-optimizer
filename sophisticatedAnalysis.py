'''
Author: James Le
Filename: sophisticatedAnalysis.py
Description: Explore the spatial behavior of the people of New York as can be inferred by examining their cab usage.
'''

# Import Packages
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import decomposition
from scipy import stats
from sklearn import cluster

matplotlib.style.use('fivethirtyeight')
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.figsize'] = (10,10)

def preprocess(data):
    '''
    Function to pre-process data and return lattitude/longtitude coordinates
    '''

    # remove obvious outliers
    allLat = np.array(list(data['pickup_latitude']) + list(data['dropoff_latitude']))
    allLong = np.array(list(data['pickup_longitude']) + list(data['dropoff_longitude']))

    longLimits = [np.percentile(allLong, 0.3), np.percentile(allLong, 99.7)]
    latLimits  = [np.percentile(allLat , 0.3), np.percentile(allLat , 99.7)]
    durLimits  = [np.percentile(data['trip_duration'], 0.4), np.percentile(data['trip_duration'], 99.7)]

    data = data[(data['pickup_latitude']   >= latLimits[0] ) & (data['pickup_latitude']   <= latLimits[1]) ]
    data = data[(data['dropoff_latitude']  >= latLimits[0] ) & (data['dropoff_latitude']  <= latLimits[1]) ]
    data = data[(data['pickup_longitude']  >= longLimits[0]) & (data['pickup_longitude']  <= longLimits[1])]
    data = data[(data['dropoff_longitude'] >= longLimits[0]) & (data['dropoff_longitude'] <= longLimits[1])]
    data = data[(data['trip_duration']     >= durLimits[0] ) & (data['trip_duration']     <= durLimits[1]) ]
    data = data.reset_index(drop=True)

    allLat = np.array(list(data['pickup_latitude']) + list(data['dropoff_latitude']))
    allLong = np.array(list(data['pickup_longitude']) + list(data['dropoff_longitude']))

    # convert fields to sensible units
    medianLat = np.percentile(allLat,50)
    medianLong = np.percentile(allLong,50)

    latMultiplier = 111.32
    longMultiplier = np.cos(medianLat*(np.pi/180.0)) * 111.32

    data['duration [min]'] = data['trip_duration']/60.0
    data['src lat [km]']   = latMultiplier  * (data['pickup_latitude']   - medianLat)
    data['src long [km]']  = longMultiplier * (data['pickup_longitude']  - medianLong)
    data['dst lat [km]']   = latMultiplier  * (data['dropoff_latitude']  - medianLat)
    data['dst long [km]']  = longMultiplier * (data['dropoff_longitude'] - medianLong)

    allLat  = np.array(list(data['src lat [km]'])  + list(data['dst lat [km]']))
    allLong = np.array(list(data['src long [km]']) + list(data['dst long [km]']))

    return allLat, allLong

def plotHist(allLat, allLong, data):
    '''
    Function to plot histograms of trip duration, latitude and longitude
    '''

    fig, axArray = plt.subplots(nrows=1, ncols=3, figsize=(13,4))

    axArray[0].hist(data['duration [min]'], 80)
    axArray[0].set_xlabel('trip duration [min]')
    axArray[0].set_ylabel('counts')
    axArray[1].hist(allLat, 80)
    axArray[1].set_xlabel('latitude [km]')
    axArray[2].hist(allLong, 80)
    axArray[2].set_xlabel('longitude [km]')

    plt.savefig('Figures/trip-duration-lat-long.png')

def plotDurationDistance(data):
    '''
    Function to plot the trip duration vs the aerial distance between pickup and dropoff
    '''

    data['log duration'] = np.log1p(data['duration [min]'])
    data['euclidian distance'] = np.sqrt((data['src lat [km]'] - data['dst lat [km]'] ) ** 2 +
                                           (data['src long [km]'] - data['dst long [km]']) ** 2)

    fig, axArray = plt.subplots(nrows=1, ncols=2, figsize=(13,6))

    axArray[0].scatter(data['euclidian distance'], data['duration [min]'], c='r', s=5, alpha=0.01)
    axArray[0].set_xlabel('Aerial Euclidian Distance [km]')
    axArray[0].set_ylabel('Duration [min]')
    axArray[0].set_xlim(data['euclidian distance'].min(),data['euclidian distance'].max())
    axArray[0].set_ylim(data['duration [min]'].min(),data['duration [min]'].max())
    axArray[0].set_title('trip Duration vs Aerial trip Distance')

    axArray[1].scatter(data['euclidian distance'], data['log duration'], c='r', s=5, alpha=0.01)
    axArray[1].set_xlabel('Aerial Euclidian Distance [km]')
    axArray[1].set_ylabel('log(1+Duration) [log(min)]')
    axArray[1].set_xlim(data['euclidian distance'].min(),data['euclidian distance'].max())
    axArray[1].set_ylim(data['log duration'].min(),data['log duration'].max())
    axArray[1].set_title('log of trip Duration vs Aerial trip Distance')

    plt.savefig('Figures/trip-duration-aerial-distance-pickup-dropoff.png')

def plotSpatialDensity(allLat, allLong, latRange, longRange, imageSize):
    '''
    Function to plot spatial density plot of the pickup and dropoff locations
    '''

    allLatInds  = imageSize[0] - (imageSize[0] * (allLat  - latRange[0])  / (latRange[1]  - latRange[0]) ).astype(int)
    allLongInds = (imageSize[1] * (allLong - longRange[0]) / (longRange[1] - longRange[0])).astype(int)

    locationDensityImage = np.zeros(imageSize)
    for latInd, longInd in zip(allLatInds, allLongInds):
        locationDensityImage[latInd, longInd] += 1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12))
    ax.imshow(np.log(locationDensityImage + 1), cmap='hot')
    ax.set_axis_off()

    plt.savefig('Figures/spatial-density-pickup-dropoff.png')

    return locationDensityImage

def plotTripCluster(data, numClusters):
    '''
    Function to cluster all 1.4 million trips to 80 stereotypical template trips and then look at the distribution of this "bag of trips" and how it changes over time.
    '''

    tripAttributes = np.array(data.loc[:,['src lat [km]','src long [km]','dst lat [km]','dst long [km]','duration [min]']])
    meanTripAttr = tripAttributes.mean(axis=0)
    stdTripAttr  = tripAttributes.std(axis=0)
    tripAttributes = stats.zscore(tripAttributes, axis=0)

    TripKmeansModel = cluster.MiniBatchKMeans(n_clusters=numClusters, batch_size=120000, n_init=100, random_state=1)
    clusterInds = TripKmeansModel.fit_predict(tripAttributes)

    clusterTotalCounts, _ = np.histogram(clusterInds, bins=numClusters)
    sortedClusterInds = np.flipud(np.argsort(clusterTotalCounts))

    plt.figure(figsize=(12,4))
    plt.title('Cluster Histogram of all trip')
    plt.bar(range(1,numClusters+1),clusterTotalCounts[sortedClusterInds])
    plt.ylabel('Frequency [counts]')
    plt.xlabel('Cluster index (sorted by cluster frequency)')
    plt.xlim(0,numClusters + 1)

    plt.savefig('Figures/cluster-histogram-trip.png')

    return meanTripAttr, stdTripAttr

def ConvertToImageCoords(latCoord, longCoord, latRange, longRange, imageSize):
    '''
    Function to image coordinates
    '''

    latInds = imageSize[0] - (imageSize[0] * (latCoord  - latRange[0])  / (latRange[1]  - latRange[0]) ).astype(int)
    longInds = (imageSize[1] * (longCoord - longRange[0]) / (longRange[1] - longRange[0])).astype(int)
    return latInds, longInds

def plotTypicalTrips(meanTripAttr, stdTripAttr, numClusters, latRange, longRange, imageSize, locationDensityImage):
    '''
    Function to show the template trips on the map
    '''

    templateTrips = TripKmeansModel.cluster_centers_ * np.tile(stdTripAttr, (numClusters, 1)) + np.tile(meanTripAttr,(numClusters, 1))

    srcCoords = templateTrips[:, :2]
    dstCoords = templateTrips[:, 2:4]

    srcImCoords = ConvertToImageCoords(srcCoords[:, 0],srcCoords[:, 1], latRange, longRange, imageSize)
    dstImCoords = ConvertToImageCoords(dstCoords[:, 0],dstCoords[:, 1], latRange, longRange, imageSize)

    plt.figure(figsize=(12,12))
    plt.imshow(np.log(locationDensityImage + 1), cmap='hot')
    plt.grid('off')
    plt.scatter(srcImCoords[1], srcImCoords[0], c='m', s=200, alpha=0.8)
    plt.scatter(dstImCoords[1], dstImCoords[0], c='g', s=200, alpha=0.8)

    for i in range(len(srcImCoords[0])):
        plt.arrow(srcImCoords[1][i], srcImCoords[0][i],
                dstImCoords[1][i]-srcImCoords[1][i],
                dstImCoords[0][i]-srcImCoords[0][i],
                edgecolor='c', facecolor='c',
                width=0.8,
                alpha=0.4,
                head_width=10.0,
                head_length=10.0,
                length_includes_head=True)

    plt.savefig('Figures/typical-trips.png')

if __name__ == '__main__':
    # Read train file
    dataDir = 'Taxi-Trip-Duration-Data/'
    taxiDB = pd.read_csv(dataDir + 'train.csv')

    # Pre-process data
    allLat, allLong = preprocess(taxiDB)

    # Plot the resulting histograms of trip duration, latitude and longitude
    plotHist(allLat, allLong, taxiDB)

    # Plot the trip Duration vs the Aerial Distance between pickup and dropoff
    plotDurationDistance(taxiDB)

    # Show the log density of pickup and dropoff locations
    imageSize = (700, 700)
    longRange = [-5, 19]
    latRange = [-13, 11]
    # Plot spatial density plot of the pickup and dropoff locations
    locationDensityImage = plotSpatialDensity(allLat, allLong, latRange, longRange, imageSize)

    # Create useful fields to be used
    pickupTime = pd.to_datetime(taxiDB['pickup_datetime'])

    taxiDB['src hourOfDay'] = (pickupTime.dt.hour * 60.0 + pickupTime.dt.minute) / 60.0
    taxiDB['dst hourOfDay'] = taxiDB['src hourOfDay'] + taxiDB['duration [min]'] / 60.0

    taxiDB['dayOfWeek'] = pickupTime.dt.weekday
    taxiDB['hourOfWeek'] = taxiDB['dayOfWeek'] * 24.0 + taxiDB['src hourOfDay']

    taxiDB['monthOfYear'] = pickupTime.dt.month
    taxiDB['dayOfYear'] = pickupTime.dt.dayofyear
    taxiDB['weekOfYear'] = pickupTime.dt.weekofyear
    taxiDB['hourOfYear'] = taxiDB['dayOfYear'] * 24.0 + taxiDB['src hourOfDay']

    # Plot trip clusters and their distribution
    numClusters = 80
    meanTripAttr, stdTripAttr = plotTripCluster(taxiDB, numClusters)

    # Show the typical trips on the map
    plotTypicalTrips(meanTripAttr, stdTripAttr, numClusters, latRange, longRange, imageSize, locationDensityImage)
