<!-- Slide number: 1 -->
# Problem Statement
Predicting the future stock trend with news sentiment classification based on the financial news articles.
Due to the rapid changes or fluctuations of stock price depends on the gains and losses of certain companies. News articles are on of the most important factors which will influence the stock market.
This project is about taking non quantifiable data such as financial news articles about a company and predicting its future stock trend with news sentiment classification. Assuming that news articles have impact on stock market, an attempt to study the relationship between news and stock trend will be carried out.

<!-- Slide number: 2 -->
# Objective
The main objective of this project is to enhance the accuracy of the model.
This can be done through better text-to-vector transformation i.e. bag of words.
Thereafter performing 5-Cross validation or 10-Cross validation for testing the model.

<!-- Slide number: 3 -->
# Proposed Model

<!-- Slide number: 4 -->
# Work Done
EDA and Data Visualization
Data Preprocessing
Construction of Bag of words
Implementation of Decision Tree Classifier
Implementation of Random Forest Classifier
Implementation of Passive Aggressive Classifier
Implementation of Support Vector Machine
Implementation of Multinomial Naïve Bayes Classifier
Implementation of Modified Multinomial Naïve Bayes Classifier
Applying 5 cross validation and 10 cross validation for testing the model
Comparing the models based on different parameters

<!-- Slide number: 6 -->
# Dataset
![image alt](https://github.com/Rogue5189/PREDICTION-OF-STOCK-VALUE-CHANGES-BASED-ON-THE-SENTIMENT-ANALYSIS-OF-NEWS-HEADLINES/blob/cd65228abc3638854a1084d29c8e05fa8c592968/pasted%20file.png)

<!-- Slide number: 7 -->
# Exploratory Data Analysis
Visualizing the data.
Label value count.
Total number of data points.

<!-- Slide number: 8 -->
# Data Preprocessing
Removing punctuations from the news headlines.
Renaming column names for ease of access.
Converting headlines to lowercase.
Joining all the top 25 news headlines to a single sentence.
Splitting the training and testing data.

<!-- Slide number: 9 -->
# Bag of words
Bag of words:
Sent1 – Amazon is a good company
Sent2 – Microsoft is better than Amazon
Sent3 – Amazon and Microsoft are good company

| Sent1 | Amazon | Good | Company |  |
| --- | --- | --- | --- | --- |
| Sent2 | Microsoft | Better | Amazon |  |
| Sent3 | Amazon | Microsoft | Good | Company |
| Word | Frequency |
| --- | --- |
| Amazon | 3 |
| Microsoft | 2 |
| Good | 2 |
| Company | 2 |
| Better | 1 |
|  | Amazon | Microsoft | Good | Company | Better |
| --- | --- | --- | --- | --- | --- |
| Sent1 | 1 | 0 | 1 | 1 | 0 |
| Sent2 | 1 | 1 | 0 | 0 | 1 |
| Sent3 | 1 | 1 | 1 | 1 | 0 |

<!-- Slide number: 10 -->
# Decision Tree Classifier
Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.
In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.
The decisions or the test are performed on the basis of features of the given dataset.
It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions.

![image alt](https://github.com/Rogue5189/PREDICTION-OF-STOCK-VALUE-CHANGES-BASED-ON-THE-SENTIMENT-ANALYSIS-OF-NEWS-HEADLINES/blob/b2f14010815267f6b6a9c18784681fc407a054c6/pasted%20file.png)

<!-- Slide number: 12 -->
# Random Forest Classifier
Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML.
Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset.
Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.


![image alt](https://github.com/Rogue5189/PREDICTION-OF-STOCK-VALUE-CHANGES-BASED-ON-THE-SENTIMENT-ANALYSIS-OF-NEWS-HEADLINES/blob/8271145bc45294e628abda313580a7adee84737a/pasted%20filey.png)
 
<!-- Slide number: 14 -->
# Passive Aggressive Classifier
Passive-Aggressive algorithms are generally used for large-scale learning. It is one of the few ‘online-learning algorithms‘. In online machine learning algorithms, the input data comes in sequential order and the machine learning model is updated step-by-step, as opposed to batch learning, where the entire training dataset is used at once.
This is very useful in situations where there is a huge amount of data and it is computationally infeasible to train the entire dataset because of the sheer size of the data. We can simply say that an online-learning algorithm will get a training example, update the classifier, and then throw away the example.

<!-- Slide number: 15 -->
# How Passive-Aggressive Works:
  Passive-Aggressive algorithms are called so because :
Passive :
If the prediction is correct, keep the model and do not make any changes. i.e., the data in the example is not enough to cause any changes in the model.
Aggressive :
If the prediction is incorrect, make changes to the model. i.e., some change to the model may correct it.

<!-- Slide number: 16 -->
# Support Vector Machine
Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.
The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.
SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine. Consider the below diagram in which there are two different categories that are classified using a decision boundary or hyperplane:

![image alt](https://github.com/Rogue5189/PREDICTION-OF-STOCK-VALUE-CHANGES-BASED-ON-THE-SENTIMENT-ANALYSIS-OF-NEWS-HEADLINES/blob/68a7ab6a7e3f04dbe14e99fdd7b41e35eb6434fe/y.png)

<!-- Slide number: 18 -->
# Multinomial Naïve Bayes Classifier
The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.


The term Multinomial Naive Bayes simply lets us know that each p(fi / c) is a multinomial distribution, rather than some other distribution. This works well for data which can easily be turned into counts, such as word counts in text.

The distribution we had been using with Naive Bayes classifier is a Gaussian distribution, so I guess we could call it a Gaussian Naive Bayes classifier.

In summary, Naive Bayes classifier is a general term which refers to conditional independence of each of the features in the model, while Multinomial Naive Bayes classifier is a specific instance of a Naive Bayes classifier which uses a multinomial distribution for each of the features.

<!-- Slide number: 20 -->
![image alt](https://github.com/Rogue5189/PREDICTION-OF-STOCK-VALUE-CHANGES-BASED-ON-THE-SENTIMENT-ANALYSIS-OF-NEWS-HEADLINES/blob/4f40f5d6132d91857a753d845dc54748d8f042a5/Screenshot_20260116_213247.png)
<!-- Slide number: 21 -->
![image alt](https://github.com/Rogue5189/PREDICTION-OF-STOCK-VALUE-CHANGES-BASED-ON-THE-SENTIMENT-ANALYSIS-OF-NEWS-HEADLINES/blob/80ae606add49b45987b2b3272db390f5c890f468/Screenshot_20260116_213642.png)

<!-- Slide number: 22 -->
# How a word is positive/negative sentiment
Let us take an example of a negatively marked and positively marked sentence:
Amazon does a lot of frauds while delivery.            negative
Flipkart delivery service is fast.		      positive
Now if we consider the first sentence then :
It will learn to associate the following words Amazon, does, lot, frauds, while, and delivery all as negative sentiments.
Obviously we don’t want Amazon or delivery or lot to be in negative sentiment, so unlearning this will require training set instances with the word Amazon, delivery or lot in them that are labeled as positive.

In the second sentence delivery is a positive sentiment word which will negate the effect of previous sentence.

<!-- Slide number: 23 -->
# Comparison of different models:
| Model | Training Time(sec) | Accuracy % (accuracy\_score) (count of correct predictions) |
| --- | --- | --- |
| Decision Tree Classifier | 108.233363151550 | 49.20634920634920 |
| Random Forest Classifier | 90.2992811203003 | 84.65608465608465 |
| Passive Aggressive Classifier | 0.35799646377563 | 85.44973544973545 |
| Support Vector Machine | 66.0001437664032 | 50.79365079365079 |
| Multinomial Naive Bayes Classifier | 0.12466573715209 | 85.41973123973545 |
| Modified Multinomial Naive Bayes Classifier | 0.12436687546542 | 51.23654944356875 |
Comparing different models based on parameters such as training time and accuracy

<!-- Slide number: 24 -->
# Insights from comparison
Training time of Multinomial Naïve Bayes and Passive Aggressive Classifier was found relatively low compared to other models.
Whereas Training time of Random forest, Decision Tree, and Support Vector Machine have a better scope to be improved.
Accuracy of Passive aggressive classifier, Random forest classifier and Multinomial classifier was found to be good enough and best as per our use case(Stock Sentiment).


We have found out that Multinomial Naïve Bayes takes the word and gives it a value which is further used to calculate the sentiment of a particular headline.
But it does not account for words who have a greater impact when grouped together i.e. “not good” will be treated as “not” and “good” but it has similar meaning as “bad”.
We have tried to make an adaptive Multinomial Naïve Bayes classification model which can group two words who have a greater meaning together when calculating the probability of a sentence.

<!-- Slide number: 26 -->


<!-- Slide number: 27 -->

<!-- Slide number: 28 -->
# How a word is positive/negative sentiment
Let us take an example of a negatively marked and positively marked sentence:
-Amazon does a lot of frauds while delivery.      -->      negative
-Flipkart delivery service is fast.		-->      positive
Now if we consider the first sentence then :
It will learn to associate the following words Amazon, does, lot, frauds, while, and delivery all as negative sentiments.
Obviously we don’t want Amazon or delivery or lot to be in negative sentiment, so unlearning this will require training set instances with the word Amazon, delivery or lot in them that are labeled as positive.
In the second sentence delivery is a positive sentiment word.

<!-- Slide number: 29 -->
# Conclusion
From the above block diagram we can conclude the following : 

We have got an accuracy of 51.23% for the model.

We got to know that when we combined 2 words then words having no meaning were more as compared to the number of words that have a greater meaning when combined together.

For now we merged only two words together but we can have a future scope of combining more than 2 words that have a greater impact as a whole.

<!-- Slide number: 30 -->
Optimizing the grouping of words in a more optimal manner according to their meaning.
Enhancing the adaptive model to achieve better accuracy.
Further optimizing the model for even better overall performance in terms of accuracy, time of training and in also when the model is working on real time data.
