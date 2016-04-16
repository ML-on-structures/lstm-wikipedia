

For the set of revisions needed for a page:

* We can have n recent revisions of that page
* We can have a window of revisions around that particular revision
* Set a parameter to decide how to get this thing

Parameters:

1. Depth (How many LSTM layers do we need)
2. Feature vector per layer (What all revision features do we use at each layer)
3. Cap on no. (M) of elements per sequence (revisions per (page or user))
4. Method to get the set of revisions (eg., for a page, should we get latest M, or should get M upto revision under concern, etc.)
5. Sequencing function feed revisions into LSTM (Do we shuffle?, or use time, or a same permutation, or any other sorting, by quality etc.) 
6. Hidden parameters of each LSTM layer

Parameters at Data extraction time:

1
3
4


Parameters when running the LSTM:

1
2
5
6


Dict structure created for usage (stored in JSON) should contain only the required features per layer.

