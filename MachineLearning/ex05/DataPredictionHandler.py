from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from collections import Counter
import operator
from IPython import embed



LABEL = 'Vote'
LABEL_INT = 'VoteInt'
LABELS = [LABEL, LABEL_INT]
SELECTED_FEATURES = [
	'Will_vote_only_large_partyInt',
	'Avg_Satisfaction_with_previous_vote',
	'Number_of_valued_Kneset_members',
	'Yearly_IncomeK',
	'Overall_happiness_score',
	'Is_Most_Important_Issue_Other',
	'Is_Most_Important_Issue_Financial',
	'Is_Most_Important_Issue_Environment',
	'Is_Most_Important_Issue_Military',
	'Is_Most_Important_Issue_Education',
	'Is_Most_Important_Issue_Foreign_Affairs',
	'Is_Most_Important_Issue_Social',
	'Avg_monthly_expense_when_under_age_21',
	'Garden_sqr_meter_per_person_in_residancy_area'
]


class DataPredictionHandler(object):
	def __init__(self, full_data, new_data):
		self.full_data = full_data
		self.new_data = new_data

	def predict(self):
		self.observe_cross_validation()
		prediction = self.predict_on_unlabel_data()
		self.observe_predicted_data(prediction)
		#self.plot_prediction(prediction)
		self.dump_perdictions()

	def observe_cross_validation(self):
		X_data = self.full_data.drop(LABELS, axis=1)[SELECTED_FEATURES]
		Y_data = self.full_data[LABEL]
		label_pred = cross_val_predict(self.classifier(), X_data.values, Y_data.values)
		print '\nClasification report:\n', classification_report(Y_data.values, label_pred, digits=2)
		print '\nConfussion matrix:\n', confusion_matrix(Y_data.values, label_pred)

	def predict_on_unlabel_data(self):
 		classifier = self.classifier()
 		classifier.fit(self.full_data.drop(LABELS, axis=1)[SELECTED_FEATURES], self.full_data[LABEL])
 		return classifier.predict(self.new_data[SELECTED_FEATURES].values)

 	def observe_predicted_data(self, prediction):
 		items = Counter(prediction).items()
 		print '\nPrediction:\n',
 		class_to_counter = sorted(items, key=operator.itemgetter(1), reverse=True)
 		total = float(self.new_data.shape[0])
 		for _class, counter in class_to_counter:
 			print '%s have %d which are %.2f%%' % (_class, counter, counter / total * 100.0)
		self.new_data[LABEL] = prediction

	def plot_prediction(self, prediction):
		items = Counter(prediction).items()
		labels, values = zip(*items)
		indexes = np.arange(len(labels))
		width = 1
		plt.bar(indexes, values, width)
		plt.xticks(indexes + width * 0.5, labels)
		plt.title('Prediction')
		plt.show()

	def classifier(self):
		return RandomForestClassifier(n_estimators=200, min_samples_split=4)

	def dump_perdictions(self):
		prediction_result = pd.DataFrame()
		prediction_result['IdentityCard_Num'] = self.new_data.IdentityCard_Num
		prediction_result['PredictVote'] = self.new_data[LABEL]
		prediction_result.to_csv('./prediction_result.csv', index=False)

