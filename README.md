# Quora-question-pair-similarity
Quora Question Pair Similarity using Machine Learning and Deep Learning

**NOTE:
download `train.csv.zip` and `test.csv.zip` form here: [Click to download dataset!](https://www.kaggle.com/c/quora-question-pairs/data)**. Extract the dataset and put jupyter notebook and the dataset in the same folder/location.

## Description
Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.(Credit: Kaggle.com)


## Problem Statement
* Identify which questions asked on Quora are duplicates of questions that have already been asked.
* This could be useful to instantly provide answers to questions that have already been answered.
* We are tasked with predicting whether a pair of questions are duplicates or not.

## Source/Useful Links

* **Source of this problem**: [Click Here](https://www.kaggle.com/c/quora-question-pairs)
* **Discussions**: [Click to see!](https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments)
* **Blog 1:** [Click to see!](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning)
* **Blog 2:** [Click to see!](https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30)
* **Kaggle Winning Solution and other approaches:** [Click to see!](https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0)


## Machine Learning Problem

### Data Overview
- Data will be in a file Train.csv
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate
- Size of Train.csv - 60MB
- Number of rows in Train.csv = 404,290

### Example Data Point
|id |qid1 |qid2 |question1 |question2 |is_duplicate|
|:--|:--|:--|:--|:--|--:|
|0 |1 |2 |What is the step by step guide to invest in share market in india?|"What is the step by step guide to invest in share market?| 0|

## Real World/Business Objectives and Constraints

* The cost of a mis-classification can be very high.
* You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
* No strict latency concerns.
* Interpretability is partially important.

## Mapping the real-world problem to a Machine Learning Problem

### Type of Machine Learning Problem
It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.

### Performance Metrics
Source: [Click to see performance merics on kaggle.com!](https://www.kaggle.com/c/quora-question-pairs#evaluation)
__Metrics__: 
* __Log-loss:__ [Click to see log-loss](https://www.kaggle.com/wiki/LogarithmicLoss)
* __Binary Confusion Matrix__

### Train and Test Construction

We build train and test by randomly splitting in the ratio of 70:30 or 80:20. whatever we choose as we have sufficient points to work with.

## Exploratory Data Analysis

### distribution of the dataset
![image](https://user-images.githubusercontent.com/32350208/123009989-61615780-d3db-11eb-9ffe-2d980afbd503.png)

### unique and repeated questions
![image](https://user-images.githubusercontent.com/32350208/123010168-abe2d400-d3db-11eb-82c1-3af12f8d5138.png)

### Log-Histogram of question appearance counts
![image](https://user-images.githubusercontent.com/32350208/123010282-ef3d4280-d3db-11eb-9263-ae93870a74d6.png)

### Basic Feature Extraction (before cleaning)
Let us now construct a few features like:
 - ____freq_qid1____ = Frequency of qid1's
 - ____freq_qid2____ = Frequency of qid2's 
 - ____q1len____ = Length of q1
 - ____q2len____ = Length of q2
 - ____q1_n_words____ = Number of words in Question 1
 - ____q2_n_words____ = Number of words in Question 2
 - ____word_Common____ = (Number of common unique words in Question 1 and Question 2)
 - ____word_Total____ =(Total num of words in Question 1 + Total num of words in Question 2)
 - ____word_share____ = (word_common)/(word_Total)
 - ____freq_q1+freq_q2____ = sum total of frequency of qid1 and qid2 
 - ____freq_q1-freq_q2____ = absolute difference of frequency of qid1 and qid2 

### Advance Feature Extraction(NLP and Fuzzy Features)
**Definition**:
- __Token__: You get a token by splitting sentence a space
- __Stop_Word__ : stop words as per NLTK.
- __Word__ : A token that is not a stop_word


**Features**:
- __cwc_min__ :  Ratio of common_word_count to min lenghth of word count of Q1 and Q2 <br>cwc_min = common_word_count / (min(len(q1_words), len(q2_words))
<br>
<br>

- __cwc_max__ :  Ratio of common_word_count to max lenghth of word count of Q1 and Q2 <br>cwc_max = common_word_count / (max(len(q1_words), len(q2_words))
<br>
<br>

- __csc_min__ :  Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2 <br> csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))
<br>
<br>

- __csc_max__ :  Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2<br>csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))
<br>
<br>

- __ctc_min__ :  Ratio of common_token_count to min lenghth of token count of Q1 and Q2<br>ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))
<br>
<br>

- __ctc_max__ :  Ratio of common_token_count to max lenghth of token count of Q1 and Q2<br>ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))
<br>
<br>
        
- __last_word_eq__ :  Check if First word of both questions is equal or not<br>last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])
<br>
<br>

- __first_word_eq__ :  Check if First word of both questions is equal or not<br>first_word_eq = int(q1_tokens[0] == q2_tokens[0])
<br>
<br>
        
- __abs_len_diff__ :  Abs. length difference<br>abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
<br>
<br>

- __mean_len__ :  Average Token Length of both Questions<br>mean_len = (len(q1_tokens) + len(q2_tokens))/2
<br>
<br>

- __fuzz_ratio__ :  https://github.com/seatgeek/fuzzywuzzy#usage
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
<br>
<br>

- __fuzz_partial_ratio__ :  https://github.com/seatgeek/fuzzywuzzy#usage
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
<br>
<br>

- __token_sort_ratio__ : https://github.com/seatgeek/fuzzywuzzy#usage
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
<br>
<br>

- __token_set_ratio__ : https://github.com/seatgeek/fuzzywuzzy#usage
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
<br>
<br>

- __longest_substr_ratio__ :  Ratio of length longest common substring to min lenghth of token count of Q1 and Q2<br>longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))


### T-SNE Visualization
![image](https://user-images.githubusercontent.com/32350208/123010897-1ea07f00-d3dd-11eb-9f52-aa243c09482a.png)

## Model Performance
* Logistic Regression is doing well.

## Credit
* [Applied AI Course:](https://www.appliedaicourse.com) Thanks for teaching Machine Learning and Deep Learning.
