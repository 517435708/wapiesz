import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_plots(df):
    counts = df['MemeLabel'].value_counts()

    t = [0, 20, 40, 60, 81]
    for k in range(len(t)-1):
        counts[t[k]:t[k+1]].plot(kind='bar')
        plt.tight_layout()
        plt.show()