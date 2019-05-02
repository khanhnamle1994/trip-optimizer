import math
import pandas as pd
import numpy as np
from aco import ACO, Graph
from plot import plot
import datetime
import pickle
import argparse
from geopy.geocoders import Nominatim
import xgboost as xgb
import pprint

filename = "../xgb_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))
geolocator = Nominatim(user_agent="aco-application")


def time_cost_between_points(loc1, loc2, passenger_count, pickup_datetime, store_and_fwd_flag=0):
    """
    Calculate the time (in minutes) between two points
    using the trained XGB model
    """
    date_list = pickup_datetime.split(' ')[0].split('-')
    my_date = datetime.date(year=int(date_list[0]), month=int(
        date_list[1]), day=int(date_list[2]))

    time_list = pickup_datetime.split(' ')[1].split(':')
    pickup_hour = int(time_list[0])
    pickup_minute = int(time_list[1])

    model_data = {'passenger_count': passenger_count,
                  'pickup_longitude': loc1['x'],
                  'pickup_latitude': loc1['y'],
                  'dropoff_longitude': loc2['x'],
                  'dropoff_latitude': loc2['y'],
                  'store_and_fwd_flag': bool(store_and_fwd_flag),
                  'pickup_month': my_date.month,
                  'pickup_day': my_date.day,
                  'pickup_weekday': my_date.weekday(),
                  'pickup_hour': pickup_hour,
                  'pickup_minute': pickup_minute,
                  'latitude_difference': loc2['y'] - loc1['y'],
                  'longitude_difference': loc2['x'] - loc1['x'],
                  'trip_distance': trip_distance_cost(loc1, loc2)
                  }

    df = pd.DataFrame([model_data], columns=model_data.keys())
    pred = np.exp(loaded_model.predict(xgb.DMatrix(df))) - 1
    return pred[0]


def trip_distance_cost(loc1, loc2):
    """
    Calculate the manhattan distance between two points using 
    polar coordinates in taxicab geometry https://en.wikipedia.org/wiki/Taxicab_geometry
    """
    return 0.621371 * 6371 * (
        abs(2 * np.arctan2(np.sqrt(np.square(
            np.sin((abs(loc2['y'] - loc1['y']) * np.pi / 180) / 2))),
            np.sqrt(1-(np.square(np.sin((abs(loc2['y'] - loc1['y']) * np.pi / 180) / 2)))))) +
        abs(2 * np.arctan2(np.sqrt(np.square(np.sin((abs(loc2['x'] - loc1['x']) * np.pi / 180) / 2))),
                           np.sqrt(1-(np.square(np.sin((abs(loc2['x'] - loc1['x']) * np.pi / 180) / 2)))))))


# Read in command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("loc_count", type=int,
                    help="specify number of locations (default is 15)")
parser.add_argument("ant_count", type=int,
                    help="specify number of ants to use (default is 10)")
parser.add_argument("g", type=int,
                    help="specify number of generations (default is 100)")
parser.add_argument("alpha", type=float,
                    help="specify alpha the relative importance of pheromone (default is 1.0)")
parser.add_argument("beta", type=float,
                    help="specify beta the relative importance of heuristic information (default is 10.0)")
parser.add_argument("rho", type=float,
                    help="specify rho the pheromone evaporation / residual coefficient (default is 0.5)")
parser.add_argument("q", type=float,
                    help="specify q the pheromone intensity (default is 10.0)")
parser.add_argument("--verbose", action="store_true",
                    help="print out each generation cost and best path")
args = parser.parse_args()

locations = []
points = []
# Read in the user specified number of lines from the test data
df = pd.read_csv("../Taxi-Trip-Duration-Data/test.csv")[:args.loc_count]
for index, row in df.iterrows():
    locations.append({
        'index': index,
        'x': row['pickup_longitude'],
        'y': row['pickup_latitude']
    })
    points.append((row['pickup_longitude'], row['pickup_latitude']))

# Build complete cost matrix based on time between points
cost_matrix = []
rank = len(locations)
for i in range(rank):
    row = []
    for j in range(rank):
        row.append(time_cost_between_points(
            locations[i], locations[j],
            1, str(datetime.datetime.now()),
            0))
    cost_matrix.append(row)

# Default values without user specifying from command line
# aco = ACO(ant_count=10, generations=100, alpha=1.0,
#           beta=10.0, rho=0.5, q=10, strategy=1)

# Pass in user arguments
aco = ACO(ant_count=args.ant_count, generations=args.g, alpha=args.alpha,
          beta=args.beta, rho=args.rho, q=args.q, strategy=2)

# Build graph with cost matrix and number of points
graph = Graph(cost_matrix, rank)
# Get results from ant colony, specify whether verbose output
path, cost = aco.solve(graph, args.verbose)

# Print out and plot final solution
print('final cost: {} minutes, path: {}'.format(cost, path))
print("final path addresses:")
addresses = []
for p in path:
    addresses.append(geolocator.reverse(
        f"{points[p][1]}, {points[p][0]}").address)
pprint.pprint(addresses)
plot(points, path)
