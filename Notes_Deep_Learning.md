

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


To manage the data:
Around a particular revision, only write the earliest work done on it compared with earliest first:
so what was n01 (1,2,3,4,5) to n12 (1,2,3,4,5), make it just n01 = 1 and n12 = 1

So for a user, with each revision, we just consider the next interaction on it compared to just the previous one.
This way the number of edges out of a user are limited to the number of revisions by the user. Just the commentary on next step of that user.

Then also use SQLite without using DAL. Also consider using SQLite in just the memory itself since it is very very fast access.

This group of stuff together should give better file sizes.

Then work on the LSTM depth structure as discussed.