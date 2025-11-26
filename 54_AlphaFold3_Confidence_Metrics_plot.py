# this module provides a simple overview of all confidence measures in an AlphaFold 3 prediction.


from itertools import chain
import json 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from typing import Dict, Optional, Tuple


# function to load JSON data from a file
def load_json_data(json_file_path):
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None



# 1, For Global confidence measures and ipTM matrix
def plot_global_confidence(confid_json_file_path, output_path):
    """
    Description
    -----------
    plot global confidence metrics, such as ipTM matrix and chain-wise ipTM/pTM scores.

    Args
    ----
    confid_json_file_path : str
        path to the JSON file containing global confidence metrics.
    output_path : str
        path to save the output plots and data files.

    Returns
    -------
    output a text file of scalar values,
    a tsv file of list values,
    and several pdf files of plots for list and 2D ndarray values.
        
    Notes
    -----
    - 1, Currently, the function of customizing output file names is not yet supported. 
    All generated figures will be saved in a single dedicated folder, 
    and the folder will be named by the system for your convenience.
    """

    global_confidence = load_json_data(confid_json_file_path)

    # extract global confidence measures
    chain_iptm = np.asarray(global_confidence['chain_iptm'], dtype=float) # list
    chain_pair_iptm = np.asarray(global_confidence['chain_pair_iptm'], dtype=float) # 2D ndarray 
    chain_pair_pae_min = np.asarray(global_confidence['chain_pair_pae_min'], dtype=float) # 2D ndarray
    chain_ptm = np.asarray(global_confidence['chain_ptm'], dtype=float) # list
    fraction_disordered = global_confidence['fraction_disordered'] # scalar 
    has_clash = global_confidence['has_clash'] # boolean
    iptm = global_confidence['iptm'] # scalar
    num_recycles = global_confidence['num_recycles'] # scalar
    ptm = global_confidence['ptm'] # scalar
    ranking_score = global_confidence['ranking_score'] # scalar

    # (1) For Scalar value, we directly print them and write them in a text file
    # Maybe we can plot them in structure viewer
    print(f"Fraction Disordered: {fraction_disordered}")
    print(f"Has Clash: {has_clash}")
    print(f"ipTM: {iptm}")
    print(f"Number of Recycles: {num_recycles}")
    print(f"pTM: {ptm}")
    print(f"Ranking Score: {ranking_score}")
    
    # write them to a text file
    with open(f"{output_path}/global_confidence_chain_SCALAR_measures.txt", "w") as f:
        f.write(f"Fraction Disordered: {fraction_disordered}\n")
        f.write(f"Has Clash: {has_clash}\n")
        f.write(f"ipTM: {iptm}\n")
        f.write(f"Number of Recycles: {num_recycles}\n")
        f.write(f"pTM: {ptm}\n")
        f.write(f"Ranking Score: {ranking_score}\n")

    # nunmerical chain index to chain id conversion
    def number_to_chain_idx(num):
        """Convert a chain index to a chain ID (A, B, C, ..., Z, AA, AB, ...)."""
        letter = ""
        while num >= 0:
            letter = chr(num % 26 + ord('A')) + letter
            num = num // 26 - 1
        return letter


    # (2) For list value, we output them as a dataframe, and plot them as a line plot or bar plot (chain id as x axis)
    # for chain_iptm or chain_ptm, we can plot them as line plot and bar plot
    x = np.arange(len(chain_iptm))
    # first line plot
    plt.figure(figsize=(10,5))
    plt.plot(chain_iptm, marker='o', color='b', alpha=0.7, label='Chain ipTM')
    plt.plot(chain_ptm, marker='o', color='r', alpha=0.7, label='Chain pTM')
    plt.title("Chain ipTM/pTM Scores")
    plt.xlabel("Chain Index")
    plt.ylabel("ipTM/pTM Score")
    plt.xticks(x, [number_to_chain_idx(i) for i in x])
    plt.legend()
    plt.grid()
    # we save the figure into a pdf file
    plt.savefig(f"{output_path}/global_confidence_chain_LIST_measures_lineplot.pdf", bbox_inches='tight')
    plt.close()

    # and bar plot
    width = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, chain_iptm, width, label='Chain ipTM', color='b', alpha=0.7)
    plt.bar(x + width/2, chain_ptm, width, label='Chain pTM', color='r', alpha=0.7)
    plt.title("Chain ipTM/pTM Scores")
    plt.xlabel("Chain Index")
    plt.ylabel("ipTM/pTM Score")
    plt.xticks(x, [number_to_chain_idx(i) for i in x])
    plt.legend()
    plt.grid()
    # we save the figure into a pdf file
    plt.savefig(f"{output_path}/global_confidence_chain_LIST_measures_barplot.pdf", bbox_inches='tight')
    plt.close()


    # write them to a tsv file
    with open(f"{output_path}/global_confidence_chain_LIST_measures.tsv", "w") as f:
        f.write("Chain_Index\tChain_ipTM_Score\tChain_pTM_Score\n")
        for i in range(len(chain_iptm)):
            f.write(f"{number_to_chain_idx(i)}\t{chain_iptm[i]}\t{chain_ptm[i]}\n")


    # (3) For 2D ndarray value, we output them as heatmap plot (chain id as x and y axis)
    import seaborn as sns
    # for chain_pair_iptm
    plt.figure(figsize=(8,6))
    sns.heatmap(chain_pair_iptm, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=[number_to_chain_idx(i) for i in range(chain_pair_iptm.shape[0])],
                yticklabels=[number_to_chain_idx(i) for i in range(chain_pair_iptm.shape[1])])
    plt.title("Chain Pair ipTM Scores Heatmap")
    plt.xlabel("Chain Index")
    plt.ylabel("Chain Index")
    # we save the figure into a pdf file
    plt.savefig(f"{output_path}/global_confidence_chain_pair_iptm_heatmap.pdf", bbox_inches='tight')
    plt.close()

    # for chain_pair_pae_min
    plt.figure(figsize=(8,6))
    sns.heatmap(chain_pair_pae_min, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=[number_to_chain_idx(i) for i in range(chain_pair_pae_min.shape[0])],
                yticklabels=[number_to_chain_idx(i) for i in range(chain_pair_pae_min.shape[1])])
    plt.title("Chain Pair PAE Min Heatmap")
    plt.xlabel("Chain Index")
    plt.ylabel("Chain Index")
    # we save the figure into a pdf file
    plt.savefig(f"{output_path}/global_confidence_chain_pair_pae_min_heatmap.pdf", bbox_inches='tight')
    plt.close()

    

# 2, For local confidence measures
def plot_local_confidence(full_json_file_path, output_path, chains: Optional[object]=None, tick_step: int = 100):
    """
    Description
    -----------
    plot local confidence metrics, such as PAE matrix and per-atom pLDDT scores.
    
    Args
    ----
    full_json_file_path : str
        path to the JSON file containing local confidence metrics.
    output_path : str
        path to save the output plots and data files.
    chains : str or list or tuple, optional, currently designed for PAE matrix plot only
        specify the chain id(s) to plot. If None, plot all chains. Default is None.
        - str: single chain id, e.g., 'A'
        - list or tuple: multiple chain ids, e.g., ['A', 'B']
    tick_step : int, optional
        step size for residue ticks on axes. Default is 100.

    Returns
    -------
    output a figure of PAE matrix for the specified chains.
    
    """

    # load local confidence data
    local_confidence = load_json_data(full_json_file_path)
    if local_confidence is None:
        raise ValueError("Failed to load local confidence data from JSON file.")
    
    # extract local confidence measures
    pae_matrix = np.asarray(local_confidence['pae'], dtype=float)  # 2D ndarray
    contact_probs = np.asarray(local_confidence['contact_probs'], dtype=float)  # 2D ndarray
    atom_chain_ids = np.asarray(local_confidence['atom_chain_ids'], dtype=str)  # list of str   
    atom_plddts = np.asarray(local_confidence['atom_plddts'], dtype=float)  # list of float
    token_chain_ids = np.asarray(local_confidence['token_chain_ids'], dtype=str)  # list of str
    token_res_ids = np.asarray(local_confidence['token_res_ids'], dtype=int)  # list of int


    # convert chains param into a hashable list
    if chains is None:
        selected_chains = None
    elif isinstance(chains, str):
        # single chain id as string, like 'A'
        selected_chains = [chains]
    else:
        # list or tuple of chain ids, like ['A','B'] or ('A','B')
        selected_chains = list(chains)

    # select token indices for the specified chains
    if selected_chains is not None:
        missing = [chain_id for chain_id in selected_chains if chain_id not in np.unique(token_chain_ids)]
        if missing:
            raise ValueError(f"Specified chains not found in data: {missing}")
        
        # create a boolean mask for selected chains, res in selected chains will be True
        mask = np.isin(token_chain_ids, selected_chains)
        idx = np.where(mask)[0] # tuple to index array

        if idx.size == 0:
            raise ValueError("No residues found for the specified chains.")
        
        # filter pae_matrix and contact_probs based on selected chains, and other token-based arrays
        # Note: we only filter token-based arrays here, atom-based arrays will be filtered later
        pae_matrix_sub = pae_matrix[np.ix_(idx, idx)]
        contact_probs_sub = contact_probs[np.ix_(idx, idx)]
        token_chain_ids_sub = token_chain_ids[idx]
        token_res_ids_sub = token_res_ids[idx]

    # if no chains specified, use all data (default)
    else:
        pae_matrix_sub = pae_matrix
        contact_probs_sub = contact_probs
        token_chain_ids_sub = token_chain_ids
        token_res_ids_sub = token_res_ids
        idx = np.arange(pae_matrix.shape[0])
        
    # compute tick positions and labels for the selected chains
    xticks_loc = []
    xticks_labels = []
    for index, res in enumerate(token_res_ids_sub):
        # Note that index is the index in the sub-matrix, while res is the residue id
        # for example, index 728 in submatrix may correspond to residue 1 in chain B
        if res == 1 or (tick_step and res % tick_step == 0):
            xticks_loc.append(index)
            xticks_labels.append(int(res))

    # 1, Plot PAE matrix for specified chains
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(pae_matrix_sub, cmap="Greens_r", origin="upper", aspect="auto")
    ax.set_title("Predicted Aligned Error (PAE) Matrix", fontsize=10)
    ax.set_xlabel("Scored Residue", fontsize=10)
    ax.set_ylabel("Aligned Residue", fontsize=10)
    ax.set_xticks(xticks_loc)
    ax.set_xticklabels(xticks_labels, fontsize=10)
    ax.set_yticks(xticks_loc)
    ax.set_yticklabels(xticks_labels, fontsize=10)

    # add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.045)
    cbar.set_label("Expected Position Error (Å)", fontsize=10)

    # if multiple chains present (either full file with >1 chain, or selected_chains with >1 chain)
    # draw small colored bars on top and left showing chain segmentation
    unique_chains = np.unique(token_chain_ids_sub)
    draw_bars = len(unique_chains) > 1
    # if chain left > 1, we will draw bars and segmentation lines
    if draw_bars:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="5%", pad=0.03)
        ax_left = divider.append_axes("left", size="5%", pad=0.03)

        # build contiguous blocks (chain_id, start, end) relative to token_chain_ids_sub order
        chain_blocks = []
        start = 0
        for i in range(1, len(token_chain_ids_sub) + 1):
            if i == len(token_chain_ids_sub) or token_chain_ids_sub[i] != token_chain_ids_sub[start]:
                chain_blocks.append((token_chain_ids_sub[start], start, i - 1))
                start = i

        # create a color map for chains, map chain ids -> integers for colors
        chain_to_int = {chain_id: i for i, chain_id in enumerate(unique_chains)}
        chain_row = np.asarray([chain_to_int[chain_id] for chain_id in token_chain_ids_sub]).reshape(1, -1)

        # use pastel colors for the top/left chain bars and slightly transparent
        cmap = plt.get_cmap("tab10", len(unique_chains))

        ax_top.imshow(chain_row, cmap=cmap, aspect="auto", alpha=0.9)
        ax_top.set_xticks([])
        ax_top.set_yticks([])

        ax_left.imshow(chain_row.T, cmap=cmap, aspect="auto", alpha=0.9)
        ax_left.set_xticks([])
        ax_left.set_yticks([])

        # draw dashed separations on main axes at block boundaries (skip boudarys at 0 and N)
        for _, s, e in chain_blocks:
            if s != 0:
                sep = s - 0.5
                ax.axvline(sep, color="k", linewidth=1, linestyle="--")
                ax.axhline(sep, color="k", linewidth=1, linestyle="--")
                # subtle white divider on small bars for visual alignment
                ax_top.axvline(sep, color="w", linewidth=1)
                ax_left.axhline(sep, color="w", linewidth=1)
        
        # annotate chain ids centered on their contiguous blocks (bold)
        for cid, s, e in chain_blocks:
            center = (s + e) / 2.0
            ax_top.text(center, 0, str(cid), ha="center", va="center",
                        fontsize=14, weight="bold", color="#222222")
            ax_left.text(0, center, str(cid), ha="center", va="center",
                         fontsize=14, weight="bold", color="#222222", rotation=90)
    
        # save figure
        plt.savefig(f"{output_path}/local_confidence_PAE_matrix_selected_chains_{'_'.join(selected_chains)}.pdf", bbox_inches='tight')
