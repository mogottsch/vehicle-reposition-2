# A Novel Vehicle Repositioning Strategy for Heterogeneous Fleets

This repository contains the implementation of a relocation algorithm for Free-Floating Vehicle Sharing Systems.

For further details concerning the theory behind our model, see ...

## Installation

Our implementation is written in python and makes use of various libraries.  
To quickly install all requirements we recommend using [conda](https://www.anaconda.com/).  
All necessary requirements can be installed with the help of the `requirements.txt` file.
To create a new conda environment with all requirements use the following command: `conda env create --file requirements.txt`.  
Make sure to select the newly created environment as your jupyter kernel.

## Project Structure

The models source code is structured in folders: `modules` and `notebooks`.  
The folder `notebooks` contains jupyter notebooks. These notebooks should be executed in the correct order, marked by their filenames prefix.  
Some code is relocated into the `modules` folder for reusability purposes.  
Before running any notebook make sure to set the `PATH_DIR_TRIPS_RAW` configuration variable in the `config.py` to the directory path of the trip data.

## Configuration

The `modules/config.py` file allows to configure some of the models hyperparameters as well as the solver that is used to solve the linear program.

## Performance

As the underlying Linear Program of our model has a large number of variables and constraints and solving Integer Linear Programs is generally NP-hard, it can take very long to solve depending on your hardware.  
General tips to improve performance (and decrease model accuracy):

- Reduce `H3_RESOLUTION` (to 6).
- Increase `PERIOD_DURATION` (to 12).
- Only relocate in certain periods, e.g. only the first one. Set `RELOCATION_PERIDOS_INDEX` to 0.
- Reduce number of (reduced) scenarios. e.g. set `N_REDUCED_SCENARIOS` to 4.
- Increase number of threads used by the solver. Set `SOLVER_OPTIONS` to `{"threads": x}`
- Discard value-at-risk. Set `beta` to `0` when calling `stochastic_program.create_model(beta=0)`.
- Decrease fleet size `factory.set_initial_allocation(small_fleet_size)`.
