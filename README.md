# Wikipedia LSTM #

This repository describes the inmplementation of MLSL specifically to Wikipedia data.

See [project page](https://sites.google.com/view/ml-on-structures) for the overall project, papers, and data. 


The implementation is independent of the language of wikipedia being used.


## Running Code ##

* In order to run the code, currently data needs to be placed in results directory under wiki name. For example, for bgwiki, unzip the data into results as results/bgwiki/
* Inside this wiki directory, there should be a file reduced_user_graph.json
* Data for astwiki can be obtained from: https://drive.google.com/file/d/0B7XE3zJNvKQXQ09RcmJWN3A2V3M/view?usp=sharing
* Then in the file graph_analysis.py, make sure that WIKINAME is same as the wiki being used
* Run ``python graph_analysis.py`` to run MLSL
* Results are stored in results/wikiname directory
