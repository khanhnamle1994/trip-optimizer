# Final Project for CSCI 633
# Biologically-Inspired Intelligent Systems

### Requirements
* Python3
* matplotlib
* numpy
* pandas
* xgboost
* geopy

### Installation
```
$ python3 -m pip install xgboost
$ python3 -m pip install geopy
```

### Usage
```
$ python3 aco_main.py -h
usage: aco_main.py [-h] [--verbose] loc_count ant_count g alpha beta rho q

positional arguments:
  loc_count   specify number of locations (default is 15)
  ant_count   specify number of ants to use (default is 10)
  g           specify number of generations (default is 100)
  alpha       specify alpha the relative importance of pheromone (default is
              1.0)
  beta        specify beta the relative importance of heuristic information
              (default is 10.0)
  rho         specify rho the pheromone evaporation / residual coefficient
              (default is 0.5)
  q           specify q the pheromone intensity (default is 10.0)

optional arguments:
  -h, --help  show this help message and exit
  --verbose   print out each generation cost and best path
```