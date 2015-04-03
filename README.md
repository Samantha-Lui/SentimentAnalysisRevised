#Sentiment Analysis on Twitter

###Summary

> This application performed a lexicon-based sentiment analysis on a sample collected from Twitter API Version 1.1. Each lexicon score are on a scale of -5 to 5 with a great number indicating a more positive sentiment and vice versa. The sentiment score of a given tweet is the total of the sentiment scores of the lexicons it contains.

> The sentiment analysis features:

> 1. A statistic of the sentiment scores by the states in the U.S..

> 2. A statistic on the hashtags included in the sample.

> 3. A summary of the sentiment lexicons associated with each of the hashtags and their frequencies.

> 4. A summary of the sentiment terms and their frequencies in the sample.

> A sample of the result can be found in results.txt in this repo.

This application was written in Python 2.7 and takes three arguments from the command line:

1. _us-states.json_ : Containing a JSON object which holds the names of the states in the U.S. as the keys and the coordinates of a list of vertices around the states as the respective values.

2. sample.txt: A text file containing the raw data collected from Twitter API.

3. _AFFINN-111.txt_: The sentiment dictionary distributed under [Open Database License (ODbL) v1.0](http://www.opendatacommons.org/licenses/odbl/1.0/) for the sentiment scores computation.

