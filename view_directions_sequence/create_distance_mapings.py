"""
This Script creates Distances Mappings of Sequences.
For every seq it creates a bounding box around it, according to it's min/max coordinates.
Then it's looks which seqs are in a certain distance of each other specified by the  the boorder arg 
"""


import numpy as np
import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--border',type=int)

def create_bounding_boxes(df):
    """
    This creates a bounding box by simply using the min,max coordinates
    """
    mins = df.groupby("sequence_key")[["easting","northing","City"]].min()
    maxs = df.groupby("sequence_key")[["easting","northing"]].max()

    maxs.columns = ["x_max","y_max"]
    mins.columns = ["x_min","y_min","City"]
    bounding_boxes = pd.concat([maxs,mins],axis=1)
    bounding_boxes = bounding_boxes[["x_min","x_max","y_min","y_max","City"]]
    return bounding_boxes


if __name__ == "__main__":

    args = parser.parse_args()
    border = args.border

    # read the Distances. For our research we settled for max 50 meters betwen seqs
    distances = pd.read_feather("max_distance_of_50.feather")
    df = pd.read_feather("prelim.feather").set_index("key")
    keys = np.concatenate(distances["keys"].values)
    # only keep images in our seqs
    df = df.loc[keys]

    bounding_boxes = create_bounding_boxes(df)

    n_citis = df["City"].nunique()
    matches = {}

    done = 0

    for pos,i in enumerate(bounding_boxes.groupby("City")):

        print(f"Now Doing City {i[0]}, {len(i[1])} seqs, {done/len(bounding_boxes) *    100:.2f}% done")
        
        val_df = i[1].iloc[:,:-1]
        done += len(val_df)
        seqs = val_df.index

        # now he have to double loop over all the sequences in a given city
        for q in tqdm.tqdm(range(len(seqs))):

            first = val_df.iloc[q].values
            hits = []

            for j in range(q +1 ,len(seqs)):

                second = val_df.iloc[j]

                # check for crossing of bounding rects
                x = (first[0] -border <= second[0]  <= first[1] + border) or (first[0] -border <= second[1]  <= first[1] + border) or (second[0] -border <= first[0] <= second[1]+border) or (second[0] -border <= first[1] <= second[1]+border) 
                y = (first[2] -border <= second[2]  <= first[3] + border) or (first[2] -border <= second[3]  <= first[3] + border) or (second[2] -border <= first[2] <= second[3]+border) or (second[2] -border <= first[3] <= second[3]+border) 

                # if there is one append the seq
                if x and y:

                    hits.append(seqs[j])

            # if we found a match save list
            if hits:
                matches[seqs[q]] = [hits]

    # save results
    matches = pd.DataFrame.from_dict(matches,orient='index')
    matches.columns = ["Mathes"]
    matches.reset_index().to_feather(f"matches_with_{border}_border.feather")            


