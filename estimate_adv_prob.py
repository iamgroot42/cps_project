import numpy as np
import os
from numpy.core.defchararray import lower
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


class AdvSuccessProb:
    def __init__(self, data, thresholds=(0.5, 2.0)):
        self.mapping = {}
        
        # Aggregate (mean-level) based on distances
        for x in data:
            self.mapping[int(x[0])] = self.mapping.get(int(x[0]), []) + [x[1]]
        
        for k, v in self.mapping.items():
            # Use means
            # Clip everything inside loss thresholds
            self.mapping[k] = np.clip(np.mean(v), thresholds[0], thresholds[1])

        all_vals = list(self.mapping.values())
        all_keys = list(self.mapping.keys())
        minv, maxv = np.min(all_vals), np.max(all_vals)

        # Scale to [0, 1] range
        for k, v in self.mapping.items():
            self.mapping[k] = (v - minv) / (maxv - minv)
    
        self.min_sup, self.max_sup = np.min(all_keys), np.max(all_keys)

    # Get probability for any specific distance
    def prob(self, x):
        if x < self.min_sup or x > self.max_sup:
            return 0

        if int(x) == x:
            if x in self.mapping:
                return self.mapping[x]
            return self.prob(x+1)
        else:
            lower_prob = self.prob(int(x))
            upper_prob = self.prob(int(x) + 1)

            ratio = x - int(x)
            prob = lower_prob * ratio + upper_prob * (1 - ratio)
            return np.clip(prob, 0, 1)


def read_file(fp):
    data = []
    with open(fp, 'r') as f:
        for line in f:
            data.append([float(x) for x in line.rstrip('\n').split(',')])
    return np.array(data)


def read_logs(folder):
    data = []
    for path in os.listdir(folder):
        data.append(read_file(os.path.join(folder, path)))
    data = np.array(data)
    return  np.reshape(data, (-1, 2))


if __name__ == "__main__":
    # Set dark background
    plt.style.use('dark_background')

    # Read log data, squeeze together
    data = read_logs("./logs")

    prob_obj = AdvSuccessProb(data)
    
    # 200 to 1600, in jumps of 0.1
    x, y = [], []
    for i in np.arange(200, 1600, 0.1):
        x.append(i)
        y.append(prob_obj.prob(i))
    
    plt.plot(x, y)
    plt.savefig("./probs.png")
    plt.clf()

    columns = [
        "Distance from landing strip",
        "Loss value for correct prediction  "
    ]
    df = pd.DataFrame(data, columns=columns)

    # Plot function
    sns_plot = sns.lineplot(data=df, x=columns[0], y=columns[1])
    sns_plot.figure.savefig("./adv_model.png")
