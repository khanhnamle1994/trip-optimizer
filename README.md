# Final Project for CSCI 633
## Biologically-Inspired Intelligent Systems

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
  loc_count   number of locations (default is 15)
  ant_count   number of ants to use (default is 10)
  g           number of generations (default is 100)
  alpha       relative importance of pheromone (default is 1.0)
  beta        relative importance of heuristic information (default is 10.0)
  rho         pheromone residual coefficient (default is 0.5)
  q           pheromone intensity (default is 10.0)

optional arguments:
  -h, --help  show this help message and exit
  --verbose   print out each generation cost and best path
```
```
$ python3 genetic_evo_main.py -h
usage: genetic_evo_main.py [-h] [--verbose] loc_count n g m c

positional arguments:
  loc_count   number of locations (default is 15)
  n           population size (default is 10)
  g           number of generations (default is 100)
  m           mutation factor (default is 0.5)
  c           crossover rate (default is 0.7)

optional arguments:
  -h, --help  show this help message and exit
  --verbose   print out each generation cost and best path
```