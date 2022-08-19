"""
This scripts clusters the image sequence of the Citys.
It uses the precalculated mathces of create_distance_mapings.py.
These have to be passed in via the --matches_df flag
"""

import numpy as np
import os
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from create_distance_mapings import create_bounding_boxes
import argparse
sns.set_theme()

parser = argparse.ArgumentParser()
parser.add_argument('--matches_df',type=str)
parser.add_argument('--output_path',type=str)

def get_cluster_clouds(cluster_to_seq,seq_to_cluster):
    """
    This function creates clouds of cluster of neighboring clusters in the cluster graph
    """

    # get a list of all collections between clusters
    cluster_connectiosn = [set([seq_to_cluster[seq] for seq in cluster_to_seq[cluster]]) for cluster in cluster_to_seq]

    clusters = []

    while cluster_connectiosn:
        # pop the head
        head = cluster_connectiosn.pop(0)

        while True:
            change = False

            for cluster in cluster_connectiosn:
                # if we find an intersection
                if len(head.intersection(cluster)):
                    head = head.union(cluster) # add clusters 
                    cluster_connectiosn.remove(cluster)
                    change = True # and redo search
                    break
            
            if not change:
                break

        clusters.append(head)

    return clusters

def cluster_city(val_df):
    """
    This functions does the main magic and clusters a city
    """
    cluster_to_seq = {}
    seq_to_cluster = {}

    uniqu_cluster_counter = 0

    # go through all the sequnces
    for key,series in val_df.iterrows():

        # if not seen before
        if key not in seq_to_cluster:
            
            # get neighboring seqs
            todo = list(series.loc["Matches"])

            # we found a new cluster
            seq_to_cluster[key] = uniqu_cluster_counter
            cluster_to_seq[uniqu_cluster_counter] = {key}

            while len(todo):
                # get first node
                head = todo.pop(0)
                # add it to the cluster regardless if is it's already in another
                cluster_to_seq[uniqu_cluster_counter].add(head)
                
                # if not labeled before, label it
                if head not in seq_to_cluster:
                    
                    seq_to_cluster[head] = uniqu_cluster_counter


                if head in val_df.index:

                    # get it's child nodes
                    nodes = val_df.loc[head,"Matches"]

                    # for all child not
                    for node in nodes:
                        # add to cluster
                        cluster_to_seq[uniqu_cluster_counter].add(node)


                        # if not expanded already
                        if node not in seq_to_cluster:
                            # expand later
                            todo.append(node)

                # trim down doubles
                todo = list(set(todo))


            uniqu_cluster_counter += 1

    return cluster_to_seq,seq_to_cluster



def plot_city_analysis(city,ax):
    # get a city
    city_df = data_df[data_df["City"]==city].copy()
    
    # get the view counts
    view_counts = {}
    for seq_key, series in city_df.groupby("sequence_key")["view_direction"]:

        view_counts[seq_key] =  [(series == i).sum() for i in  ["Forward",'Sideways',"Backward"]]

    view_counts = pd.DataFrame.from_dict(view_counts,orient="index",columns=["Forward",'Sideways',"Backward"])


    df = create_bounding_boxes(city_df)
    cluster_ids = pd.DataFrame.from_dict(clusters_by_city[city],orient="index")

    df["Forward_Counts"] = view_counts.loc[df.index,"Forward"]
    df["Sideways_Counts"] = view_counts.loc[df.index,"Sideways"]
    df["Backward_Counts"] = view_counts.loc[df.index,"Backward"]

    # set clusters
    df["Cluster"] = [cluster_ids.loc[seq,0] if seq in cluster_ids.index else pd.NA  for seq in df.index]

    max_c = df["Cluster"].max()

    # if not clustered before it a "renegade" sequence out alone
    for i,seq in enumerate(df[df["Cluster"].isna()].index):

        df.loc[seq,"Cluster"] = max_c + i + 1


        #print(f"{seq} is it's own cluster: {max_c + i + 1}")

    # get cluster
    city_df["Cluster"] = city_df["sequence_key"].apply(lambda x: df.loc[x,"Cluster"])

    # from here on it's simply plotting
    samples = city_df.sample(frac = 1)
    samples.easting -= samples.easting.min()
    samples.northing -= samples.northing.min()

    max_lim = max(samples.easting.max(),samples.northing.max())
    diff = samples.easting.max() - samples.northing.max()

    if diff > 0:
        samples.northing += diff/2
    else: 
        samples.easting -= diff/2

    for c in sorted(samples.Cluster.unique()):

        j = samples[samples["Cluster"]==c]
        ax.scatter(j.easting,j.northing,label=c,alpha=0.5)

    ax.set_ylim([-2000,max_lim + 2000])
    ax.set_xlim([-2000,max_lim + 2000])
    
    m2km = lambda x, _: f'{x/1000:g}'
    ax.xaxis.set_major_formatter(m2km)
    ax.yaxis.set_major_formatter(m2km)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),title="Cluster")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.set_title(f"{city.capitalize()} with {samples.Cluster.nunique()} Clusters",fontsize=18)

    return city_df, ax
 



if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.output_path,exist_ok=True)

    # read dataframes
    distances = pd.read_feather("max_distance_of_50.feather")
    data_df = pd.read_feather("prelim.feather").set_index("key")

    keys = np.concatenate(distances["keys"].values)
    data_df = data_df.loc[keys]

    matches = pd.read_feather(args.matches_df)
    
    try:
        matches.rename(columns={"Mathes":"Matches"},inplace=True)
    except:
        pass
    matches.set_index("index",inplace=True)
    citys = data_df.groupby("sequence_key")["City"].min()
    matches["City"]=citys.loc[matches.index]


    n_seqs_per_city = data_df.groupby("City")["sequence_key"].nunique()

    clusters_by_city = {}

    for city,val_df in matches.groupby("City"):

        val_df = val_df.copy()

        # cluster city
        cluster_to_seq,seq_to_cluster = cluster_city(val_df)
        # cluster clusters
        clusters = get_cluster_clouds(cluster_to_seq,seq_to_cluster)

        city_match = {}

        # save which seq belongs to which cluster
        for n, cluster in enumerate(clusters):

            keys = set()

            for sub_cluster in cluster:
                
                keys = keys.union(cluster_to_seq[sub_cluster])

            for key in keys:
                
                assert key not in city_match

                city_match[key] = n

        clusters_by_city[city] = city_match

        if len(city_match)!=n_seqs_per_city.loc[city]:
            print(f"City {city} has {len(clusters)} clusters. Missing {n_seqs_per_city.loc[city]-len(city_match)}")
        else:
            print(f"City {city} has {len(clusters)} clusters.")

    dfs = []

    # for every city
    for city in tqdm.tqdm(sorted(data_df["City"].unique()), desc="Creating Plots"):
        fig, ax = plt.subplots(figsize=(10,10))

        #plots
        city_df, ax =  plot_city_analysis(city,ax)

        dfs.append(city_df)

        plt.savefig(f"{args.output_path}/{city}.jpg",dpi = 500)


    clustered_df = pd.concat(dfs)
    # save clustered info
    clustered_df.reset_index().to_feather(f"{args.output_path}/clustered_df.feather")
    
    assert len(clustered_df)==len(data_df), "Some Data was Lost"


