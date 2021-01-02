from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd


def find_best_k(df, target_cols, suffix, max_k=10):
    """
    Runs Kmeans clustering on target columns and outputs sse and data for elbow plotting

    :param df: pandas DataFrame
    :param target_col: column or columns to be used for K-means clustering
    :param suffix: Suffix to main title, usually target_col
    :return: dict of k:inertia values and a pandas df with new cluster labels
    """
    sse={}
    data = df[target_cols].values.reshape(-1, 1)
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
        #data["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_

    fig = plt.figure(figsize=(24, 10))
    plot = plt.plot(list(sse.keys()), list(sse.values()))
    xlabel = plt.xlabel("Number of clusters", fontsize=16, fontweight='bold')
    ylabel = plt.ylabel("Inertia", fontsize=16, fontweight='bold')
    xticks = plt.xticks(fontsize=16, fontweight='bold')
    title = plt.title(f'Selecting K-value based on inertia: {suffix}', fontsize=30, fontweight='bold')

    return sse

def create_labels(df, target_cols, k):
    '''

    :param df: pandas DataFrame
    :param target_cols: column to be used for kmeans calculations, use list if more than one
    :param k: best k found from find_best_k
    :return: new df with cluster labels
    '''
    if len(target_cols) == 1:
        data = df[target_cols].values.reshape(-1, 1)
    else:
        data = df[target_cols]
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    df["clusters"] = kmeans.labels_

    df_new = df.groupby('clusters')[target_cols].mean().reset_index()
    df_new = df_new.sort_values(by=target_cols, ascending=True).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df, df_new[['clusters', 'index']], on='clusters')
    df_final = df_final.drop(['clusters'], axis=1)
    df_final = df_final.rename(columns={"index": 'clusters'})

    table = df_final.groupby('clusters')[target_cols].describe()

    return table, df_final

def box_results(df, cat_col, cont_col, title):
    '''
    :param df: pandas DataFrame
    :param cat_col: categorical column of interest (x-axis)
    :param cont_col: continuous column of interest (y-axis)
    :param title: Main title of plot
    :return: boxplot of results
    '''

    fig = plt.figure(figsize=(24, 10))
    plot = sns.boxplot(x=df[cat_col], y=df[cont_col], hue=df[cat_col])
    xlabel = plt.xlabel(f"{df[cat_col].name[0].upper() + df[cat_col].name[1:]}", fontsize=16, fontweight='bold')
    ylabel = plt.ylabel(f"{df[cont_col].name[0].upper() + df[cont_col].name[1:]}", fontsize=16, fontweight='bold')
    xticks = plt.xticks(fontsize=16, fontweight='bold')
    yticks = plt.yticks(fontsize=16, fontweight='bold')
    legend = plt.legend(fontsize='xx-large')
    title = plt.title(title, fontsize=30, fontweight='bold')

def twoD_viz(df, xcol, ycol):
    '''
    Plotting function that provides 2d visualization to selected cols from pandas DF

    :param df: pandas DataFrame that includes cluster labels
    :param xcol: column along x-axis
    :param ycol: column along y-axis
    :return: visualization of plotted results
    '''

    X = df[xcol]
    y = df[ycol]
    colors = df['clusters']
    fig, ax = plt.subplots(1, 1, figsize=(24, 10))
    plt.scatter(X, y, c=colors, s=50, cmap='viridis', edgecolor='black')
    # fmt = '${x:,.0f}'
    # tick = ticker.StrMethodFormatter(fmt)
    # ax.yaxis.set_major_formatter(tick)
    xticks = plt.xticks(fontweight='bold', fontsize=16)
    yticks = plt.yticks(fontweight='bold', fontsize=16)
    xlabel = plt.xlabel(f"{df[xcol].name[0].upper() + df[xcol].name[1:]}", fontweight='bold', fontsize = 24)
    ylabel = plt.ylabel(f"{df[ycol].name[0].upper() + df[ycol].name[1:]}",  fontweight='bold', fontsize = 20)

