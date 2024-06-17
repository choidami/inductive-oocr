import matplotlib.pyplot as plt
import numpy as np

colors = ['royalblue', 'forestgreen', 'gold', 'tomato', 'purple']

def grouped_bar_plot(data, title):
    #   Data has structure
    #   {coinA: {cat_1: 0.3, cat_2: 0.4, cat_3: 0.3}}
    labels = list(data.keys())
    categories = list(data[labels[0]].keys())  # Assuming all coins have the same categories (X, Y, Z)
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    for i, category in enumerate(categories):
        category_data = [data[coin][category] for coin in labels]
        ax.bar(x + i*width - width, category_data, width, label=category, color=colors[i % len(colors)])

    plt.title(title, loc="left")
    ax.set_ylabel('Token probability')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.show()
