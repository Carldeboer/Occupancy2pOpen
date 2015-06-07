# Occupancy2pOpen
This python program inputs different bigwig genome browser tracks representing genome-wide accessibility or nucleosome occupancy, and uses a GMM to classify regions into occupied and unoccupied regions, outputting P(open) to a bigwig file and displaying a graph of the classification data.

Note that this program is extremely sensitive to the inputs provided - sometimes it is better to not include a track than to include one with inferior data. 

Requirements:
* python 2.7
* wigToBigWig from KentTools on you PATH
* python libraries: numpy, scipy, many others...
