Final Project: Investigate Fraud from Enron Email
=================================================

1.	Summarize goal of project and how machine learning is useful in accomplishing it.

	The goal of the project was to identify 'persons of interest, or POIs' in the Enron fraud case based on email and financial data. Machine learning allows us to employ a series of statistical models to identify patterns that otherwise would be much more difficult to find. Specifically, this dataset included 146 rows, or people, 18 of whom had previously been identified as persons of interest. For each person, there were 21 features concerning their financial information and email correspondence.

	Some outliers had to be removed from the dataset. For example, the row named 'Total' summed the data from each person, and one row was not a person at all, The Travel Agency in the Park. Also, each person's email address appeared as one of the features, so that had to be removed, and many datapoints appeared as 'NaNs'. Those had to be converted to 0. I removed feature outliers by converting the data to a Pandas dataframe and using the `.drop()` method. For the 'NaNs', I used the `.replace()` method.

	Machine learning allowed us to identify the most relevant features associated with the 'POIs' and then run those through a classifier algorithm to make predictions about which members were POI's. With sklearn metrics, we were then able to measure how effective our predictions were, based on precision and recall scores.

2.	Features used in the POI identifier and the selection process used to pick them.

	I did not use feature selection in my final algorithm. Instead I used PCA. I passed all the features to it, except the one outlier ('email_address'). PCA does not remove features, but instead assigns weights to the features based on relevance.

	```
	  Singular Value Weights for PCA Components (Unlabeled):
	  [  4.24447275e+01   2.22500512e+01   1.38160609e+01   1.08875849e+01
	     8.88939752e+00   8.18926445e+00   7.63405574e+00   6.69677337e+00
	     3.45370433e+00   2.97126529e+00   2.75511471e+00   2.22200191e+00
	     1.91319556e+00   1.74788910e+00   1.46025159e+00   1.22578446e+00
	     8.75772519e-01   5.82017169e-01   7.60290714e-02   2.03078269e-03]
	```

	I tried two different scalers, MinMaxScaler and StandardScaler. StandardScaler provided the best results. Scaling was necessary to normalize the data because we had two very different types of data: financial data and email correspondence. Also, the financial data varied significantly in amount from one feature to the next. Scaling allowed for an "apples to apples" comparison (so to speak) among the different types of data.

	My overall best estimator was a Pipeline using StandardScaler, PCA, and LogisticRegression. I engineered a new feature based on the ratio involving the number of messages involving each individual and a POI, i.e., total of all messages from-person-to-poi, to-person-from-poi, and shared-poi divided by total-messages involving that person. The rationale for this was that presumably POIs would share a higher percentage of messages with one another compared to non-POIs.

3.	Algorithm used; others that I tried; how model performance differed.

	My overall best estimator was a Pipeline using StandardScaler, PCA, and LogisticRegression:

	```
	  clf:  Pipeline(memory=None,
	       steps=[('scaling', StandardScaler(copy=True, with_mean=True, with_std=True)), ('dim_red', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
	    svd_solver='auto', tol=0.0, whiten=False)), ('CLF', LogisticRegression(C=100000000000000000000L, class_weight='balanced',
	            ...,
	            random_state=42, solver='liblinear', tol=1e-10, verbose=0,
	            warm_start=False))])
	```

	I tried many, many combinations of classifiers and parameters before I finally found one that got a high enough recall score. I used `class EstimatorSelectionHelper` created by Panagiotis Katsaroumpas to help identify the best classifier. Katsaroumpas' code can be found [here](http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/). I modified the code to suit my purposes. I added `StratifiedShuffleSplit` to the process, passed Pipelines to it, rather than just classifiers, and added a new dictionary (`best_estimators`) to keep the best performing parameters, along with `mean_score` for each Pipeline tested. Then I called the `max()` function on the `best_estimators` dictionary to get the overall best performing estimator and assigned that to `clf` to be used in `tester.py`.

	Below is an example of different models ranked by mean `f1` score. These are not by far all the models I tested. I tried about a thousand different combinations before I found one that achieved a high enough recall score. Other models included DecisionTreeClassifier, AdaBoostClassifier, GradientBoostingClassifier, and SVM. I also tried a variety of scaling, feature selection, and dimension reduction methods. Examples: `SelectKBest`, `RFE`, `PCA`, `KernelPCA`, and more.

	```
	  estimator             min_score mean_score max_score  std_score
	  0  LogisticRegression         0   0.337431  0.888889  0.165247
	  1  DecisionTreeClassifier     0   0.258988  0.833333  0.172876
	```

4.	What tuning parameters means; what can happen if it's not done well; how I tuned the parameters of my algorithm; what parameters I tuned.

	Tuning parameters involves trying different combinations of parameters to find the most effective combination. If parameter tuning is not done well, it can cause a few problems. First, if you do it manually, it can be a very tedious process: adjust, test, re-adjust, re-test, etc. If you do it in an unsupervised way, say by passing parameters to a Pipeline, it can throw an error if, for instance, you specify a number of features to test that does not fall within the range of components specified. Another problem can be that it may cause your program to run for a very long time if you specify too large a number of parameters to test.

	I used GridsearchCV inside of a function, which allowed me to iterate through various Pipelines. The Pipelines included a variety of selectors, dimension reduction methods, and classifiers along with their associated parameters. After trying out all the different combinations, the function returned a dictionary of the best performing estimator for each Pipeline. I then called the `max()` method on the dictionary to get the overall best performing estimator.

	At first, I just radndomly threw numbers into the parameters for the Pipelines and hoped for the best. This did not prove to be particularly effective. My precision and recall scores would not go above about .222. Then I began to take a more systematic approach. I began trying to isolate the best values for key parameters. Then once those were identified, I would hold those values constant and try to identify the best values for other parameters one at a time.

	For example, initially the best performing classifier I found was GradientBoostingClassifier. To tune the parameters, I followed the process described in ["Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python"](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/) by Aarshay Jain. First, I set initial estimates on a few parameters and then began tuning max_depth and num_samples_split. Once I found optimal values there, I held those constant and tuned min_samples_leaf. When I found the optimal value there, I tuned max_features. Finally, I moved to tuning subsample and then began tuning the learning rate and making proportional inverse adjustments to the number of estimators.

	Unfortunately, after all this, the best precision and recall score I got was .29 for each. Frustrated, after having spent more time than I'd like to admit on this, I finally broke down and sought "Live Help" on the Udacity site. Thankfully, the mentor suggested a few classifiers that I had not tried. I was able to plug those classifiers into my functions and with just a little bit of tuning, was able to achieve precision and recall above .3.

	My takeaway message from this project is that to do this well, you have to both really understand the nature of the data and have a thorough knowledge of the myriad classifiers, selctors, and dimension reduction methods, so that you can choose the correct Pipeline to suit the data.

5.	What validation is; a classic mistake one can make if it's done wrong; how I validated my analysis.

	Validation involves dividing your dataset into training and testing splits so that a portion can be used to train the algorithm and a portion to test it. If you were to test your algorithm using the same subset of data that it was trained on, it would be impossible to attest to the validity of the score. In fact, this is a classic mistake that is often made, causing an overfit of the data. In doing so, you get a high accuracy score, but it cannot be trusted. It would return low precision and recall scores.

	In my algorithm, I used StratifiedShuffleSplit, which returns stratified, randomized folds. I used 100 folds in my tuning and testing and set the test size to include 30% of the data.

	```
	  cv = StratifiedShuffleSplit(y, 100, test_size=0.3, random_state = 42)
	```

6.	Two evaluation metrics and average performance for each; an interpretation of the metrics.

	The final precision, recall, and 'F1' scores for my algorithm are listed below. Taken together, precision and recall measure the overall effectiveness of the algorithm, where precision relates to the number of false-positive idnetifications and recall relates to the number of false-negative idnetifications.

	For this particular project, a high precision means the algorithm is identifying a good number of POIs correctly with few incorrect identifications (i.e., few false-positives). A high recall means that the algorithm is identifying a good number of POIs correctly without missing many actual POIs (i.e., low false-negatives). The 'F1' score is the harmonic mean of the two.

	```
	  Precision: 0.37044    Recall: 0.49750 F1: 0.42467
	  Total predictions: 15000  True positives:  995    
	  False positives: 1691   False negatives: 1005   
	  True negatives: 11309
	```

	My algorithm's recall score is slightly better than the precision score, which is explained by the fewer false-negatives than false_positives. In other words, the algorithm is identifying more people as POIs who are not POIs, than it is failing to identify ones who are.

