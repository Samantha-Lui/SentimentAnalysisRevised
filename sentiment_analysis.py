"""
This application performs a lexicon-based sentiment analysis on a set of tweets using
a lexicon dictionary called AFFIN-111. The lexicon scores are on a scale of -5 to 5 with
a great number indicating a more positive sentiment and vice versa.

This script was written in Python 2.7. To run, enter at the console:
python sentiment_analysis.py us-states.json yourOwnTweetFile.txt AFINN-111.txt.

A sample result is provided in Results.txt.
"""

import multiprocessing as mp
import sys
import json
import fileinput
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string


# A model of the states in the U.S. represented by a series of polygons.
# keys: 2-letter abbreviations of the states in the U.S..
# values: the set of vertices of the polygon in which the state inscribed.
statesVertices = {}

# The sentiment lexicons dictionary.
# columns: sentiment lexicons, score, numWords.
sentimentDF = pd.DataFrame()

# The set of tweets that were from the U.S. and had nonempty text field.
# columns: text(normalized), state, score
tweetsDF = pd.DataFrame()

# Hashtags included in the tweets.
# keys: Hashtags included in the tweets.
# values: Indices in tweetsDF of which include the hashtags.
hashtags = {}

# Sentiment lexicons involved in the tweets.
# keys: Sentiment lexicons involved in the tweets.
# values: Indices in tweetsDF of which include sentiment lexicons.
lexicons = {}

##
# This group of functions loads and preprocesses the set of tweets for analysis.
##
def parseStates(stream):    
	"""
	Create a model of the states in the U.S. represented by a series of polygons.
	
	Extract the name of the states and the list of coordinates of vertices of the 
	inscribing polygon from the json object in the specified stream, and store the 
	data in statesVertices.
 
	@param the stream connecting a file containing the pertinent data.
	@type file input stream.
	"""
	states = stream.read()
	statesJ = json.loads(states)
	for i in range(len(statesJ['features'])):
		statesVertices[statesJ['features'][i]["properties"]["state"]] = \
		statesJ['features'][i]["geometry"]["coordinates"][0][0]
				
def tidyTweets(stream):
	"""
	Crop the tweets which come from U.S. and have content in the text. 

	First the function filters the set of tweets for those that were sent 
	within the U.S. and have nonempty text content. Then it normalizes the
	text content and determines the states at which the tweets came from.
	Also, the hashtags included and the indices of their originated tweets
	are recorded.
	
	@param the stream connecting a file containing the tweets.
	@type file input stream.
	"""
	rawTweets = stream.readlines() 
	rawTweets = filter(isTargetTweet, rawTweets)

	# Extract data from the tweets.
	pool = mp.Pool(processes=4)
	data = [pool.apply(getData, args=(t,)) for t in rawTweets]
	# The normalized text content.
	tweetsDF['content'] = [d[0] for d in data]
	# The right side of the equation determines the state in which the tweet came from.
	tweetsDF['state'] = [inState(c) for c in [center(bb) for bb in [d[1] for d in data]]]
	# The hashtags included in the tweet.
	htags = [d[2] for d in data]

	# Record the hashtags and the indices of their originated tweets.
	for i in range(len(htags)):
		if len(htags[i])>0:
			for ht in htags[i]:
				if not ht in hashtags.keys():
					hashtags[ht] = [i]
				else:
					hashtags[ht].append(i)	
	
def isTargetTweet(twt):
	"""
	Check if the specific tweet is a piece of valid data for our analysis.

	A valid tweet is one that was sent from the U.S. and has nonempty text content.
	
	@param the tweet in question.
	@type json
	
	@return validity of the tweet.
	@rtype boolean
	"""
	tweet = json.loads(twt)
	if 'text' in tweet.keys() and 'place' in tweet.keys() and \
        not tweet['place'] == None and tweet['place'] ['country_code'] == 'US':
		return True
	return False
	
def getData(twt):
	"""
	Extract data from the specific tweet.
	
	The data for the tweet is represented by 
	the tuple (content, bounding box, hashtags)
	where content is the normalized text content,
	       bounding box is a piece of geodata,
	       hashtages are the hashtages included in the tweet.
	
	@param the tweet in question.
	@type json.
	@return a tuple (content, bounding box, hashtags).
	@rtype (string, list, list).
	"""
	tweet = json.loads(twt)
	text = normalize(tweet['text'].encode('utf-8'))
	# A list of the vertices of the box
	coords = tweet['place'] ['bounding_box']['coordinates'][0]
	hashtags = []
	if 'entities' in tweet.keys() and len(tweet['entities']['hashtags'])>0:
		htags = tweet['entities']['hashtags']
		for htag in htags:
			ht = htag['text'].encode('utf-8')
			hashtags.append(ht.lower())
	return (text,coords,hashtags)
		
def normalize(txt):
	"""
	Normalize the text content of a tweet.
	
	Tasks included: white spaces, digits, punctuation and (customized) stopwords removal
	and decapitalization.
	
	@param the text content of a tweet.
	@type string
	@return a normalized text content.
	@rtype string.
	"""
	# Remove extra spaces.
	text = txt.strip()
	# Words to be excluded from the stopwords since they are essential for our analysis.
	negations = set(('no', 'nor', 'not'))
	punctuation = set(string.punctuation)
	digits = set(string.digits)
	# A customized set of stopwords.
	myStopwords = set.union(set.union(set(stopwords.words('english')), punctuation), digits) - negations
	# Remove stopwords and decapitalize the remaining words.
	text = ' '.join([word.lower() for word in text.split() if word not in myStopwords])
	return text
	
def inState(pt):
	"""
	Determine which state the specified point locates.

	@param the coordinates of the center of the bounding box 
	       of the tweet's location.
	@type list
	@return the two-letter abbreviation of the state in which the 
	specified point locates if a state is detected and `undetermined`
	otherwise.
	@rtype string
	"""
	x = pt[0]
	y = pt[1]

	for key in statesVertices.keys():
		xcoords = [] # List of x-coordinates of the vertices
		ycoords= [] # List of y-coordinates of the vertices
		for vertex in statesVertices[key]:
			xcoords.append(vertex[0])
			ycoords.append(vertex[1])
		n = len(xcoords)
		l = n-1
		for k in range(n):
			if ycoords[k]<y and ycoords[l]>=y or ycoords[l]<y and ycoords[k]>=y:
				z = xcoords[k] + (y-ycoords[k])*(xcoords[l]-xcoords[k])/(ycoords[l]-ycoords[l]-ycoords[k])
				if z < x:
					return key
			l=k   
	return 'undetermined'

def center(pts):
	"""
	Compute the coordinates of the center of bounding box of 
	the tweet.

	@param the specified list of coordinates of vertices.
	@type list
	@return coordinates of the center of the bounding box.
	@rtype [x-coordinate, y-coordinate]
	"""

	if len(pts) < 4:
		return pts
	p1 = pts[0]
	p2 = pts[2]
	x1 = p1[0]
	y1 = p1[1]
	x2 = p2[0]
	y2 = p2[1]   
	return [(x1+x2)/2, (y1+y2)/2] 
	
	
##
# This group of functions performs the sentiment analysis on the preprocessed dataset.
##
def loadLexicons(stream):
	"""
	Create a sentiment lexicons dictionary.
	
	Extracts the sentiment terms and their scores from
	the specified file and counts the number of words in each term.
	
	@param the stream connecting to the sentiment lexicon file.
	@type file input stream
	"""
	sfile = stream.readlines()
	terms = []
	scores = []
	for line in sfile:
		term, score  = line.split("\t")
		terms.append(str(term))
		scores.append(int(score))
	sentimentDF['term'] = terms
	sentimentDF['score'] = scores
	sentimentDF['numWords'] =  sentimentDF["term"].apply(lambda x: x.count(" ") + 1)

def sentStat():
	"""
	Summarize the sentiment analysis.
	
	Perform a statistic on the sentiment scores on the tweets 
	and displays the result.
	
	@return the summary of the sentiment analysis.
	@rtype string.
	"""
	tweetsDF['score'] = [sent_score(tweetsDF['content'][i],i) for i in range(len(tweetsDF))]
	stat = pd.DataFrame(tweetsDF.groupby(['state']).agg(['count','mean','std']))
	results = '\n\n******** STATISTIC OF SENTIMENT SCORE BY STATE ********\n'
	results += str(stat)
	results += '\n***********************************************************\n\n'
	return  results

def sent_score(text,i):
	"""
	Compute the total sentiment score from a tweet and record for the lexicons.
	
	Starting from the longest sentiment terms, the negation and the terms themselves are 
	checked against the specified text. If there is a match, the matched term is removed
	from the text. Moreover, the function takes a record of the sentiment terms and 
	the index of the tweet in tweetDF in the lexicons dictionary when appropriate.
	
	@param the normalized text content of a tweet.
	@type string
	@param the index in tweetDF of the tweet in question.
	@type int
	@return the total sentiment score from the specified text.
	@rtype numeric
	"""

	total = 0 # Score of a tweet.
	split_tweet = text.split(' ')
	n = len(split_tweet)
	
	# Searches for the 3-word terms(the longest terms) and their negations.
	span = 3
	start = 0
	while start <= n-span :
		space = ' '
		seq = (split_tweet[start:start+span])
		phrase = space.join(seq)
		if phrase in list(sentimentDF.term[sentimentDF.numWords==3]):
			if start > 0 and split_tweet[start-1]=="not":
				# Add the sentiment term and the index to the lexicons dictionary.
				tweetLexicons('not'+phrase,i)
				total -= int(sentimentDF.score[sentimentDF.term==phrase])
				del split_tweet[start-1:start+span]
				n = len(split_tweet)
				start -= 2
			else:
				tweetLexicons(phrase,i)
				total += int(sentimentDF.score[sentimentDF.term==phrase])
				del split_tweet[start:start+span]
				n = len(split_tweet)
				start -= 1
		else:
			start += 1		
    
	# Searches for the 2-word terms and their negations.
	span = 2
	start = 0
	while start <= n-span :
		space = ' '
		seq = (split_tweet[start:start+span])
		phrase = space.join(seq)
		if phrase in list(sentimentDF.term[sentimentDF.numWords==2]):
			if start > 0 and split_tweet[start-1]=="not":
				tweetLexicons('not'+phrase,i)
				total -= int(sentimentDF.score[sentimentDF.term==phrase])
				del split_tweet[start-1:start+span]
				n = len(split_tweet)
				start -= 2
			else:
				tweetLexicons(phrase,i)
				total += int(sentimentDF.score[sentimentDF.term==phrase])
				del split_tweet[start:start+span]
				n = len(split_tweet)
				start -= 1
		else:
			start += 1
			
	# Searches for the 1-word terms and their negations.
	span = 1
	start = 0
	while start <= n-span :
		space = ' '
		seq = (split_tweet[start:start+span])
		phrase = space.join(seq)
		if phrase in list(sentimentDF.term[sentimentDF.numWords==1]):
			if start > 0 and split_tweet[start-1]=="not":
				tweetLexicons('not'+phrase,i)
				total -= int(sentimentDF.score[sentimentDF.term==phrase])
				del split_tweet[start-1:start+span]
				n = len(split_tweet)
				start -= 2
			else:
				tweetLexicons(phrase,i)
				total += int(sentimentDF.score[sentimentDF.term==phrase])
				del split_tweet[start:start+span]
				n = len(split_tweet)
				start -= 1
		else:
			start += 1
	return total
	
def tweetLexicons(lex,i):
	"""
	Add a lexicon and the index in tweetDF of the tweet when the term
	came from to the lexicons dictionary.
	
	@param the sentiment lexicon.
	@type string.
	@param the index in tweetDF of the tweet when the term came from.
	@type int
	"""
	if lex in lexicons.keys():
		lexicons[lex].append(i)
	else:
		lexicons[lex] = [i]
	summary_file = open(sys.argv[3], "r")
	
##
# Overall summary on the sentiment analysis.
##
def summary(sent, tag, lex):
	"""
	Gather up the summaries for the sentiment scores,
	hashtags and lexicons involved.

	The summaries are written to 'Results.txt'.
	
	@param the sentiment score summary.
	@type string.
	@param the hashtag summary.
	@type string.
	@param the sentiment lexicon summary.
	@type string.
	"""
	summary = open('Results.txt','w')
	print >> summary, sent + tag + lex
	summary.close()

def tagReport():	
	"""
	Summarize the hashtags found in the tweets.
	
	Perform a statistic on the sentiment scores on the 
	hashtags and display the sentiment lexicons associated with them.
	
	@return the summary of the hashtags found in the tweets.
	@rtype string.
	"""
	tagDF = pd.DataFrame()
	tagDF['tag'] = hashtags.keys()
	tagDF['count'] = [len(hashtags[tag]) for tag in tagDF['tag']]
	tagDF['mean'] = [np.mean(tweetsDF.score[hashtags[tag]]) for tag in tagDF['tag']]
	tagDF['std'] = [np.std(tweetsDF.score[hashtags[tag]]) for tag in tagDF['tag']]
	results = '******** HASHTAG REPORT ********\n'
	results += str(tagDF)
	results += '\n\n'
	
	# Create a reverse dictionary on the lexicons dictionary.
	inv_lexicons = {}
	for key in lexicons.keys():
		for i in lexicons[key]:
			if not str(i) in inv_lexicons.keys():
				inv_lexicons[str(i)] = [key]
			else:
				inv_lexicons[str(i)].append(key)
	
	# Display the sentiment lexicons(if any) associated with each hashtag.
	results += 'The following shows the hashtags found in the tweets'
	results += ' and their associated sentiment lexicons:\n'
	results += 'Note: Some hashtags may not be associated with any lexicons in AFFINN-111.\n\n'
	for tag in hashtags.keys():
		results += '-- ' + tag + ' --\n'
		sentLexicons = []
		for i in hashtags[tag]:
			if str(i) in inv_lexicons.keys():
				for lex in inv_lexicons[str(i)]:
					sentLexicons.append(lex)
		results += ', '.join(sentLexicons)
		results += '\n\n'

	results +=  '***********************************************************\n\n'
	return results
	
def lexReport():
	'''
	Summarize the sentiment lexicons found in the tweets.
	
	@return the summary of the sentiment lexicons found in the tweets.
	@rtype string.
	'''
	lexDF = pd.DataFrame()
	lexDF['lexicon'] = lexicons.keys()
	lexDF['count'] = [len(lexicons[lex]) for lex in lexDF['lexicon']]
	results = '******** SENTIMENT LEXICON REPORT ********\n'
	results += str(lexDF)
	results +=  '\n\n'
	results +=  '***********************************************************\n'
	return results
	
	
def main():
	state_file = open(sys.argv[1], "r")
	tweet_file = open(sys.argv[2],"r")
	sent_file = open(sys.argv[3], "r")
    
	parseStates(state_file)
	tidyTweets(tweet_file)
	loadLexicons(sent_file)
	summary(sentStat(), tagReport(), lexReport())
	state_file.close()
	tweet_file.close()
	sent_file.close()
    

if __name__ == '__main__':
	main()


