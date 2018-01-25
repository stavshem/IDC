from sklearn.cluster import KMeans
from collections import Counter
from IPython import embed
import operator


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

class CoalitionFormingHandler(object):
	def __init__(self, new_data):
		self.new_data = new_data
		self.k = 15
		self.classifier = KMeans(n_clusters=15, random_state=0)

	def train(self):
		self.classifier.fit(self.new_data[SELECTED_FEATURES])
		self.observe_votes_for_cluster()

	def observe_votes_for_cluster(self):
		self.new_data['cluster'] = self.classifier.labels_
		counter = Counter(self.classifier.labels_)
		cluster_to_votes = {}
		for cluster in self.new_data.cluster.unique():
			max_votes_percentage = (counter[cluster] / len(self.new_data)) * 100
			votes_for_party = []
			cluster_records = self.new_data[self.new_data.cluster == cluster]
			parties_in_largest_cluster = cluster_records.Vote.unique()
			for vote in parties_in_largest_cluster:
				votes_for_party.append([vote, len(cluster_records[cluster_records.Vote == vote])])
			votes_for_party = sorted(votes_for_party, key=operator.itemgetter(1), reverse=True)
			cluster_to_votes[cluster] = votes_for_party
		for i, cluster in enumerate(cluster_to_votes, 1):
			print 'Cluster %s: %s' % (i, cluster_to_votes[cluster])
		return cluster_to_votes

