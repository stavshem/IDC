import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from sklearn import preprocessing
from DataPrepartionHandler import DataPrepartionHandler
from DataPredictionHandler import DataPredictionHandler
from CoalitionFormingHandler import CoalitionFormingHandler 

def load_data():
	return [
		pd.read_csv('./ElectionsData-full.csv', header=0),
		pd.read_csv('./ElectionsData_Pred_Features.csv', header=0),
	]

def main():
	full_data, new_data = load_data()
	DataPrepartionHandler(full_data, new_data).prepare()
	DataPredictionHandler(full_data, new_data).predict()
	CoalitionFormingHandler(new_data).train()

main()

embed()

