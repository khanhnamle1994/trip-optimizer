'''
Author: James Le
Filename: basicAnalysis.py
Description: Analyze trip duration and pickup times from the data.
'''

# library import
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def clean(data):
    '''
    Simple pre-processing/cleaning
    '''
    # Since there are less than 10k rows with anomalies in trip_duration (in common sense), we can safely remove them
    duration_mask = ((data.trip_duration < 60) | # < 1 min
             (data.trip_duration > 3600 * 2)) # > 2 hours
    print('Anomalies in trip duration, %: {:.2f}'.format(data[duration_mask].shape[0] / data.shape[0] * 100))

    data = data[~duration_mask]
    data.trip_duration = data.trip_duration.astype(np.uint16)
    print('Trip duration in seconds: {} to {}'.format(data.trip_duration.min(), data.trip_duration.max()))

    # Drop trips with passenger count = 0, since there are only 17 of them
    print('Empty trips: {}'.format(data[data.passenger_count == 0].shape[0]))
    data = data[data.passenger_count > 0]

    # Convert this feature into categorical type
    data.store_and_fwd_flag = data.store_and_fwd_flag.astype('category')

    # month (pickup and dropoff)
    data['mm_pickup'] = data.pickup_datetime.dt.month.astype(np.uint8)
    data['mm_dropoff'] = data.dropoff_datetime.dt.month.astype(np.uint8)

    # day of week
    data['dow_pickup'] = data.pickup_datetime.dt.weekday.astype(np.uint8)
    data['dow_dropoff'] = data.dropoff_datetime.dt.weekday.astype(np.uint8)

    # day hour
    data['hh_pickup'] = data.pickup_datetime.dt.hour.astype(np.uint8)
    data['hh_dropoff'] = data.dropoff_datetime.dt.hour.astype(np.uint8)

def PickupPlot1(data):
    '''
    Pickup time distribution by hour-of-day
    '''
    plt.figure(figsize=(12,2))

    data = data.groupby('hh_pickup').aggregate({'id':'count'}).reset_index()
    sns.barplot(x='hh_pickup', y='id', data=data)

    plt.title('Pick-ups Hour Distribution')
    plt.xlabel('Hour of Day, 0-23')
    plt.ylabel('No of Trips made')

    plt.savefig('Figures/pickups-hour-distribution.png')

def PickupPlot2(data, dow_names):
    '''
    Pickup time distribution by day-of-week
    '''
    plt.figure(figsize=(12,2))

    data = data.groupby('dow_pickup').aggregate({'id':'count'}).reset_index()
    sns.barplot(x='dow_pickup', y='id', data=data)

    plt.title('Pick-ups Weekday Distribution')
    plt.xlabel('Trip Duration, minutes')
    plt.xticks(range(0,7), dow_names, rotation='horizontal')
    plt.ylabel('No of Trips made')

    plt.savefig('Figures/pickups-weekday-distribution.png')

def PickupPlot3(data, dow_names):
    '''
    Pickup heatmap of day-of-week vs. hour-of-day
    '''
    plt.figure(figsize=(12,2))
    sns.heatmap(data=pd.crosstab(data.dow_pickup, data.hh_pickup, values=data.vendor_id, aggfunc='count', normalize='index'))

    plt.title('Pickup heatmap, Day-of-Week vs. Day Hour')
    plt.ylabel('Weekday')
    plt.xlabel('Day Hour, 0-23')
    plt.yticks(range(0,7), dow_names[::-1], rotation='horizontal')

    plt.savefig('Figures/pickup-heatmap-day-of-week-vs-hour.png')

def TripDurationPlot1(data):
    '''
    Trip duration distribution in minutes
    '''
    plt.figure(figsize=(12,3))

    plt.title('Trip Duration Distribution')
    plt.xlabel('Trip Duration, minutes')
    plt.ylabel('No of Trips made')
    plt.hist(data.trip_duration / 60, bins=100)

    plt.savefig('Figures/trip-duration-distribution.png')

def TripDurationPlot2(data, dow_names):
    '''
    Trip duration based on hour-of-day vs. weekday
    '''
    plt.figure(figsize=(12,2))
    sns.heatmap(data=pd.crosstab(data.dow_pickup, data.hh_pickup, values=data.trip_duration/60, aggfunc='mean'))

    plt.title('Trip duration heatmap (Minutes), Day-of-Week vs. Day Hour')
    plt.ylabel('Weekday')
    plt.xlabel('Day Hour, 0-23')
    plt.yticks(range(0,7), dow_names[::-1], rotation='horizontal')

    plt.savefig('Figures/trip-duration-heatmap.png')

if __name__ == '__main__':
    # Load train data
    taxiDB = pd.read_csv(filepath_or_buffer='Taxi-Trip-Duration-Data/train.csv', engine='c', infer_datetime_format=True, parse_dates=[2,3])

    # Clean data
    clean(taxiDB)

    # Let's add some additional columns to speed-up calculations dow names for plot mapping
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Visualization for Pick-Up feature
    PickupPlot1(taxiDB)
    PickupPlot2(taxiDB, dow_names)
    PickupPlot3(taxiDB, dow_names)

    # Visualization for Trip Duration feature
    TripDurationPlot1(taxiDB)
    TripDurationPlot2(taxiDB, dow_names)
