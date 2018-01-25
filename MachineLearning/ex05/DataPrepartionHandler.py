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


class DataPrepartionHandler(object):
	def __init__(self, full_data, new_data):
		self.full_data = full_data
		self.new_data = new_data
		self.datasets = [new_data, full_data]

	def prepare(self):
		self.observe_columns_naming_mismatch()
		self.fix_columns_naming_mismatch()
		self.observe_missing_values()
		self.transfrom_categorial_to_numeric()
		self.one_hot_encode('Most_Important_Issue')
		self.impute_missing_values()
		self.scale_values()

	def observe_columns_naming_mismatch(self):
		new_data_columns = set(self.new_data.columns.values)
		full_data_columns = set(self.full_data.columns.values)
		print 'New data is missing columns:\n%s' % ('-' * 25)
		print "\n".join((full_data_columns - new_data_columns))

	def fix_columns_naming_mismatch(self):
		rename_columns = {
			"X.Of_Household_Income": "%Of_Household_Income",
			"X.Time_invested_in_work": "%Time_invested_in_work",
			"X._satisfaction_financial_policy": "%_satisfaction_financial_policy",
			"Financial_balance_score_.0.1.": "Financial_balance_score_(0-1)"
		}
		self.new_data.rename(index=str, columns=rename_columns, inplace=True)

	def observe_missing_values(self):
		print '\nMissing Values Counter:\n%s' % ('-' * 25)
		print self.new_data.isnull().sum()
		#self.new_data.isnull().sum().plot(kind='bar', title='Missing Values Counters')
		#plt.show()

	def impute_missing_values(self):
		for dataset in self.datasets:
			float_columns = self.float_columns(dataset)
			categorical_columns = self.categorical_columns(dataset)
			for column in categorical_columns:
				dataset[column].fillna(dataset[column].value_counts().idxmax(), inplace=True)
			dataset[float_columns] = dataset[float_columns].fillna(dataset[float_columns].mean(), inplace=True)

	def scale_values(self):
		for dataset in self.datasets:
			for column in self.float_columns(dataset):
				range_size = dataset[column].max() - dataset[column].min()
				dataset[column] = (dataset[column] - dataset[column].min()) / range_size

	def transfrom_categorial_to_numeric(self):
		for dataset in self.datasets:
			for column in self.categorical_columns(dataset):
				new_column = '%sInt' % column
				values_range = range(dataset[column].nunique())
				dataset[new_column] = dataset[column].astype('category').cat.rename_categories(values_range).astype(float)

	def one_hot_encode(self, column):
		for dataset in self.datasets:
			for category in dataset[column].unique():
				new_column = 'Is_%s_%s' % (column, category)
				dataset[new_column] = (dataset[column] == category).astype(int)

	def float_columns(self, dataset):
		return dataset.select_dtypes(include=['float64']).columns

	def categorical_columns(self, dataset):
		return set(dataset.select_dtypes(include=['object']).columns) - set(LABEL)
