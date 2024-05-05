import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    # Read in data after preprocessing
    data = pd.read_csv('statistics_output.csv')

    # Generate a strip plot for each variable
    for i in data.columns:
        if (i == 'Website'):
            continue
        plt.figure(figsize=(10,5))
        sns.stripplot(data=data, x=i, y='Website', jitter=False)
        plt.title(label="Variation in the variable: "+i)
        plt.show()
