# wwv

 Sabastian Fernandes | NJIT Applied Physics Undergraduate | Summer 2023 - Current

* This repository contains many useful tools that can be used to process and visualize data from your Grape V1 low-IF receiver node.

  The the grape.py file contains two developed objects, as well as some helper functions that perform smaller operations but improve overall organization and modularization.

 1. grape class

    A grape object can be created for any properly formatted timeseries .csv file originating from a Grape V1. It contains properties from the header section of the .csv file, as well as 3 dynamically accessible arrays containing time, frequency, and relative power data in various forms. Each module in the grape class is capable of producing a different visualization of the timeseries data contained within the grape object.
    
 3. grapeHandler class

    A grapeHandler object can be created to handle any number of preloaded grape objects, each containing their own day's worth of timeseries frequency / power data. This object allows for analysis to be simultaneously performed over an arbitrary span of Grape V1 timeseries data.
    
