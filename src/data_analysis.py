import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlations(train_data):
    plt.figure(figsize=(12,8))
    sns.heatmap(train_data.corr(), cmap="flare", annot=True)
    plt.show()

def continuous_univariate_analysis(train_data, column):
    analysis_data = train_data[[column]].fillna(-1)
    fig, axs = plt.subplots(1,2)
    axs[0].hist(analysis_data, color="darkblue")
    axs[1].boxplot(analysis_data)
    plt.show()
    
def categorical_univariate_analysis(train_data, column):
    analysis_data = train_data[column].fillna(-1)
    fig, axs = plt.subplots(1,1)
    axs.bar(height = analysis_data.value_counts()/analysis_data.shape[0], x = analysis_data.unique(), color="darkblue")
    axs.set_xticks(analysis_data.unique())
    plt.show()
    
def multivariate_analysis(train_data, column):
    plt.scatter(x=train_data[column], y=train_data["disease_degree"])
    plt.show()