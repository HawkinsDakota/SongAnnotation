from pandas import DataFrame
import seaborn
import matplotlib
import matplotlib.pyplot as plt

def plot_syllable_distribution(syllables, save_file=None, plot_file=None):

    unique_syllables = list(set(syllables))
    counts = [0]*len(unique_syllables)
    syllable_dict = {each: i for i, each in enumerate(unique_syllables)}
    for each in syllables:
        index = syllable_dict[each]
        counts[index] += 1
    d = {'Syllable': unique_syllables,
         'Count': counts}
    df = DataFrame(data=d, columns=list(d.keys()))
    df = df.sort_values(by='Count')
    x_labels = [each.split('_')[1] for each in df['Syllable']]

    if save_file is not None:
        df.to_csv(save_file, index=False)
    seaborn.set()
    seaborn.set_style("darkgrid", {"axes.facecolor": ".9"})
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, sharex=False,
                                   figsize=(10, 6))

    count_plot = seaborn.barplot(x='Syllable', y='Count', data=df,
                                 orient='v', ax=ax1)

    seaborn.distplot(df['Count'],
                     hist_kws=dict(cumulative=True),
                     kde_kws=dict(cumulative=True),
                     ax=ax2)

    plt.sca(ax1)
    plt.xticks(df.index.values, x_labels, rotation='vertical')
    plt.tick_params(axis='x', which='both', labelsize=8)
    plt.ylabel('Count')
    plt.autoscale(tight=True)

    plt.sca(ax2)
    ax2.set_xlim([0, max(df['Count'].values)])
    plt.ylabel('Cumulative Percentage')
    plt.suptitle('Syllable Count Distribution')

    if plot_file is not None:
        plt.savefig(plot_file)
    else:
        plt.show()
    plt.close()

def plot_confusion(confusion, plot_file=None):
    for i in range(confusion.shape[0]):
        confusion.iloc[i, :] = confusion.iloc[i, :]/sum(confusion.iloc[i, :])
    seaborn.set_style('darkgrid', {'axes.facecolor': '0.9'})
    fig, ax = plt.subplots(figsize=(12, 10))
    cbar_ax = fig.add_axes([0.905, 0.3, 0.05, 0.3])
    heat = seaborn.heatmap(confusion, cmap='viridis',
                          cbar_kws={'label': 'Prediction'},
                          ax=ax,
                          cbar_ax=cbar_ax)
    plt.sca(ax)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.tick_params(axis='both', which='both', labelsize=8)
    if plot_file is not None:
        plt.savefig(plot_file)
    else:
        plt.show()
    plt.close()
