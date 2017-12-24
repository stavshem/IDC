import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt



train = pd.read_csv("./processed_train.csv", header=0)

selected_features_typed = [
	'Number_of_valued_Kneset_members',
	'Yearly_IncomeK',
	'Overall_happiness_score',
	'Avg_Satisfaction_with_previous_vote',
	'Is_Most_Important_Issue_Military',
	'Is_Most_Important_Issue_Other',
	'Is_Most_Important_Issue_Foreign_Affairs',
	'Will_vote_only_large_party_Int',
	'Garden_sqr_meter_per_person_in_residancy_area',
	'Weighted_education_rank'
]



kfold = KFold(n=len(train), n_folds=4, shuffle=True)
X = train[selected_features_typed].values
Y = train.Vote.values

classifiers = [
	DecisionTreeClassifier(max_depth=10, criterion='entropy'),
	Perceptron(),
	LinearSVC(random_state=0),
	KNeighborsClassifier(),
	RandomForestClassifier()
]

classfier_to_score = {}
for classifier in classifiers:
	scores = []
	for k, (train_index, test_index) in enumerate(kfold):
		classifier.fit(X[train_index], Y[train_index])
		scores.append(classifier.score(X[test_index], Y[test_index]))
	classfier_to_score[classifier] = np.mean(scores)


for classifier, avg_score in classfier_to_score.iteritems():
	print "Avg score of %s is %f" % (type(classifier).__name__, avg_score)


selected_classifiers = [classifier[0] for classifier in sorted(classfier_to_score.iteritems(), key=lambda x:-x[1])[:3]]

test = pd.read_csv("./processed_test.csv", header=0)
X_test = test[selected_features_typed]
y_true = test['Vote']
classfier_to_score = {}
for classifier in selected_classifiers:
	y_pred = classifier.predict(X_test)
	score = accuracy_score(y_true, y_pred)
	classfier_to_score[classifier] = score
	print "accuracy_score of %s is %f" % (type(classifier).__name__, score)


best_classifier = max(classfier_to_score)
predicted_votes = best_classifier.predict(X_test)
party_to_votes = Counter(predicted_votes)
winning_party = party_to_votes.most_common(1)[0][0]
votes_per_party = {party: votes / float(len(predicted_votes)) * 100 for party, votes in party_to_votes.iteritems()}ֿֿֿ

for k, v in votes_per_party.iteritems():
	print "%s will win %d % votes" % (k, v)

test['prediction'] = best_classifier.predict(X_test)
pd.DataFrame(test).to_csv("predicted_test.csv")


conf = confusion_matrix(y_true, predicted_votes)
plt.imshow(conf, cmap='binary', interpolation='None')
plt.show()


