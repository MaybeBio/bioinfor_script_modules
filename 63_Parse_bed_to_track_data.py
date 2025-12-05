# the function defined here is modified to support 1D track data visualization alongside 2D matrix
from ast import Store
from typing import Literal, Optional, List, Union, Dict, Any
import os
import pandas as pd
from matplotlib.colors import ListedColormap, to_hex
import matplotlib.pyplot as plt


# 1, transfer the bed data into what we need for 1D track plotting
def parse_bed_to_track_data(
        bed_file: str,
        track_name: str,
        track_type: str = "line",
        value_type: Literal["int", "float", "str"] = "float",
        is_categorical: bool = False,
        color: Union[str, Dict[str, str]] = "orange"
) -> Dict[str, Any]:
    """
    Description
    -----------
    Parse a BED file to extract 1D track data for visualization.
    Helper function to convert a BED-like file into the track_data dictionary format.
    
    Args
    ----
        bed_file (str): Path to the BED-like (csv file more precisly) file containing 1D track data.
        track_name (str): Name of the track to be used in visualization.
        track_type (str): Type of the track, either "line" or "bar". Default is "line".
        value_type (str): Type of the value in the "value" column (generally the third column), this can be numeric type like "float" or "int", or 
"str" for categorical data. Default is "float".
        is_categorical (bool): Whether the track data is categorical. Default is False.
        color (Union[str, Dict[str, str]]): Color for the track plot. Single value for numeric data, dict for categorical data mapping category to color.

    Notes        
    -----
    - 1, This function is designed to parse BED files for 1D track data visualization.
    Originally, we have 3 ideas:
        - a, Use Formal formatted file like bed or json to store 1D track data, and we do parsing work in main code
        - b, Use a simple dictionary format to directly provide 1D track data, and we need to do parsing work in advance, 
        and then we use the parsed data in main code, that is easy for us to implement the plotting function
        - c, Combine both a and b, we parse bed/json files using additional functions, and we then use the parsed data in main code
        to do the simple plotting work in our own way
    - 2, We assume the bed file:
        - a, has four columns: chain_id(can add your own), start, end, value
        - b, 0-based indexing
        - c, tab-separated values (tsv)
        - d, has a header row
    - 3, If is_categorical=False: Expects a single color string (e.g. "orange").
    If is_categorical=True: Expects a colormap name (e.g. "tab10") OR a dictionary {category: color}.
    - 4, for simplicity, we just treat all str values as categorical values.
    that means, if value_type is str, we always set is_categorical=True
    - 5, chain_value has the same index as bed file, that means, if bed file is 0-based indexing, chain_value is also 0-based indexing, same for 1-based indexing
    - 6, We design the data stuucture returned as: A MULTI-LIST of Track instead of Chain (the data returned will be grouped by track name instead of chain id)
    """

    # First, we load the bed-like file(tsv)
    track_df = pd.read_csv(bed_file, sep="\t", header=0)   
            
    # here, we assume the bed file has four columns: chain_id, start, end, value
    # col0=Chain, col1=Start, col2=End, col3=Value

    tracks_to_return = []


    track_dict = {}
    all_categories = set()

    # Group by Chain ID
    for chain_id, group in track_df.groupby(track_df.columns[0]):
        # Here we have a problem, how do we know the length of the chain?
        # Strategy: First store it as an interval list, and then expand it later when aligning with the actual structure length in the drawing function.
        # Alternative: Store it directly as a sparse dictionary here, in the format {residue_index: value}
        
        # here we use the second strategy
        chain_vals = {}
        
        # note that group is still a dataframe
        # and for dataframe we iterate rows using iterrows()
        for _, row in group.iterrows():
            # note that _ is the index, row is the actual row data, index is useless here
            start = int(row.iloc[1])
            end = int(row.iloc[2])

            # numerical value, like disorder score
            if value_type == "float":
                value = float(row.iloc[3])
            elif value_type == "int":
                value = int(row.iloc[3])
            else:
                # str type
                value = str(row.iloc[3])
                # categorical value, like hydrophobic, polar, charged
                if is_categorical:
                    all_categories.add(value)
                # non categorical str value, we just keep it as str, like numeri

                # Actually, non-categorical and categorical str values is difficult to distinguish here
                # For example, domain annotation, there maybe so many different domain names, we are afraid that we do not have so many colors to map, 
                # and so many colors may make the legend messy 
                # ⚠️ BUt, for simplicity, we just treat all str values as categorical values
                # that means, if value_type is str, we always set is_categorical=True

            # then we fill in the track_dict
            for i in range(start, end + 1):
                # we assume here the bed range is closure [start, end]
                chain_vals[i] = value
        
        # finally, we add this chain's data to track_dict
        # Now we have {res: bed value} -> chain_id -> track 
        track_dict[chain_id] = chain_vals 

    # For color settings, we will deal with it further
    final_color = color
    if is_categorical:
        # first, we will see how many categories we have
        unique_cats = sorted(list(all_categories))

        # Case1:
        # we already what category the annotation track will have, so we directly provide a color map
        # for example, we have three categories: "Hydrophobic":red, "Polar":blue, "Charged":green
        # {category: color}
        if isinstance(color, dict):
            final_color = color
        
        # Case2:
        # we do not provide a color map, cause category logic handled above means that we do not know what categories we will have before parsing the bed file
        # we may get so many different categories that there is no need to map them so clearly
        # so here we may just input a colormap name, and we will generate colors for each category
        elif isinstance(color, str):
            # for example, we input a color panel
            try:
                cmap = plt.get_cmap(color, len(unique_cats))
                # then we generate {category:hex color} dict
                # cause cmap(i) returns an RGBA tuple, we need to convert it to hex string
                final_color = {
                    cat: to_hex(cmap(i)) for i, cat in enumerate(unique_cats)
                }

            except ValueError:
                # if the colormap name is invalid, we just use default colors, like "tab10"
                print(f"Warning: Colormap '{color}' not found. Using 'tab10'.")
                cmap = plt.get_cmap("tab10", len(unique_cats))
                final_color = {cat: to_hex(cmap(i)) for i, cat in enumerate(unique_cats)}

            
    # finally, we construct the track_data dictionary
    return {
        "track_name": track_name,
        "track_type": track_type,
        "color": final_color,
        "is_categorical": is_categorical,
        "track_data": track_dict
    }


