# 可视化mmCIF结构残基互作图
# 并且标注其中的每一条chain id的名称

# 1, 暂时不考虑track relay
# 可以不考虑 _get_aligned_track_data以及相关代码

import os
import numpy as np
from typing import Optional, List, Union, Dict, Any
from Bio.PDB import MMCIFParser, Polypeptide
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from alphafold3_seqvis_toolkit.utils.track_utils import parse_bed_to_track_data

def contact_map_vis(
    mmcif_file: str,
    chains: Optional[Union[str, List[str]]] = None,
    out_path: Optional[str] = None,
    cmap: str = "RdBu_r", # Reversed RdBu so Red is close (contact), Blue is far
    track_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None # list supported
):
    """
    Description
    -----------
    Generate a contact map (distance matrix) for a single structure from an mmCIF file.
    Uses representative atoms: CA for Protein, C1' for Nucleic Acids, First Atom for others.
    
    Args
    -----
        mmcif_file (str): Path to the mmcif file.
        chains (str | List[str], optional): Chain ID(s) to include. 
                                            e.g., "A" or ["A", "B"]. 
                                            If None, all chains are included.
        out_path (str, optional): Path to save the output plot (e.g., "/data2").
        vmax (float): Maximum distance (Angstrom) for colorbar scaling. Default is 95th percentile of distances.
        cmap (str): Colormap to use. Default is "RdBu_r" (Red=Close, Blue=Far).
        track_data (Dict[str, Any], optional): Additional data to overlay on the contact map.
        track_data (dict/list, optional): 1D feature track data.  
            Format:
            [{
                "track_name": "IDR Score",
                "data": {"A": [0.1, ...], "B": [0.5, ...]}, # Lists must match residue count
                "color": "orange",
                "ylim": (0, 1),
                "track_type": "line" # or 'bar'
            }]
    
    Notes
    ------
    - 1, We currently support one AND list additional track_data overlay.
    - 2, The 1D track data should be prepared externally in some kind certain fomrat, see utils/track_utils.py for details.
    - 3, When not providing 1D track data, we will not plot any overlays.
    """

    job_name = os.path.splitext(os.path.basename(mmcif_file))[0].split("_")[1] 
    def _load_representative_atoms(mmcif_path, target_chains=None):
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('struct', mmcif_path)
        # AF3 usually has model 0
        model = structure[0]

        coords = []
        chain_labels = [] 
        res_types = [] # Track type for info              
        
        # Normalize target_chains
        if isinstance(target_chains, str):
            target_chains = {target_chains}
        elif isinstance(target_chains, (list, tuple)):
            target_chains = set(target_chains)
        
        found_chains = set()

        for chain in model:
            if target_chains is not None and chain.id not in target_chains:
                continue
            
            for res in chain:
                # Skip water 
                if res.id[0] == "W": continue   
                # Note that HETATM residues are included here, cause Zn2+ is HETATM, so we don't skip them
                # but if we want to skip other HETATMs, we can add more conditions here.
                # like: if res.id[0].startswith("H_"): continue
 
                rep_atom = None      
                rtype = "Unknown"

                # 1. Protein -> CA
                if Polypeptide.is_aa(res, standard=True):
                    if 'CA' in res:
                        rep_atom = res['CA']
                        rtype = "Protein"
                
                # 2. Nucleic Acid -> C1' (Distinguish DNA/RNA)
                elif "C1'" in res:
                    rep_atom = res["C1'"]
                    rname = res.resname.strip()
                    if rname in ['DA', 'DT', 'DG', 'DC']:
                        rtype = "DNA"
                    elif rname in ['A', 'U', 'G', 'C']:
                        rtype = "RNA"
                    else:
                        rtype = "Nucleic" # Fallback for modified bases
                
                # 3. Others (Ligands/Ions) -> First Atom
                else:
                    atoms = list(res.get_atoms())
                    if len(atoms) > 0:
                        rep_atom = atoms[0]
                        rtype = "Ligand"

                if rep_atom:
                    coords.append(rep_atom.get_coord())
                    chain_labels.append(chain.id)
                    res_types.append(rtype)
                    found_chains.add(chain.id)

        if not coords:
            msg = f"No valid atoms found in {mmcif_path}."
            if target_chains: msg += f" (Searched for chains: {target_chains})"
            raise ValueError(msg)

        return np.array(coords, dtype=np.float32), chain_labels, sorted(list(found_chains))


    # We need 1D track data support here 
    def _get_aligned_track_data(track_cfg, chain_labels_list):
        """
        Description
        ------------
        Align 1D track data to the residues in the structure.
        Aligns track data (sparse dict) to the flat list of residues in the structure.
        
        Args
        ----
            track_cfg (dict): Configuration for the track data.
            chain_labels_list (list): List of chain IDs corresponding to each residue (rep atom).
        
        Notes
        -----
        - 1, The track_cfg is 0-based index !
        """
        
        # check first
        if not track_cfg or "track_data" not in track_cfg:
            return None
        
        full_data = []
        # our data
        raw_data = track_cfg["track_data"]

        # we traverse all residues in the structure
        current_chain = None
        res_counter = 0

        # Align data, we calculate residue index per chain
        for cid in chain_labels_list:
            if cid != current_chain:
                current_chain = cid
                res_counter = 0 # Note that track_cfg is 0-based index!
            else:
                res_counter += 1

            # default value is nan
            val = np.nan

            # check the raw_data
            if cid in raw_data:
                chain_data = raw_data[cid]
                # get bed value, default is nan
                val = chain_data.get(res_counter, np.nan)

            # now we get the value for this residue
            full_data.append(val)
        # full_data now contains aligned values for all residues/rep atoms in the structure, whether the chain is present in track_data or not
        # that means, full_data is bed value aligned to chains in config_track, and nan for missing data
        # value column in track data can be numerical or categorical (str), so we just keep it 
        return np.asarray(full_data, dtype=object)



    # 1. Load Data
    coords, chain_labels, loaded_chains = _load_representative_atoms(mmcif_file, chains)
    N = coords.shape[0] # number of tokens/representative atoms/residues
    print(f"Loaded {N} tokens from chains: {loaded_chains}")

    # 2. Compute Distance Matrix
    # shape (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(15, 12))
    
    im = ax.imshow(
        dist_matrix,
        cmap=cmap,
        origin="upper",
        vmin=0,
        interpolation="nearest" 
    )
    
    # Create divider for existing axes instance
    divider = make_axes_locatable(ax)
    
    # Append axes to the top and left for chain bars
    ax_top = divider.append_axes("top", size="5%", pad=0.03)
    ax_left = divider.append_axes("left", size="5%", pad=0.03)
    
    # Append axes for colorbar to the right
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    # Add Colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Distance (Å)", fontsize=12)

    # Prepare chain blocks and colors
    chain_blocks = []
    start = 0
    for i in range(1, N + 1):
        if i == N or chain_labels[i] != chain_labels[start]:
            chain_blocks.append((chain_labels[start], start, i - 1))
            start = i
            
    unique_chains = sorted(list(set(chain_labels)))
    chain_to_int = {cid: i for i, cid in enumerate(unique_chains)}
    chain_row = np.array([chain_to_int[c] for c in chain_labels]).reshape(1, -1)
    
    # Use tab20 for distinct chain colors
    if len(unique_chains) <= 20:
        cmap_chains = plt.get_cmap("tab20", len(unique_chains))
    else:
        colors = plt.get_cmap("gist_rainbow")(np.linspace(0, 1, len(unique_chains)))
        cmap_chains = ListedColormap(colors)
    
    # Plot chain bars
    ax_top.imshow(chain_row, cmap=cmap_chains, aspect="auto", alpha=0.9)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    
    ax_left.imshow(chain_row.T, cmap=cmap_chains, aspect="auto", alpha=0.9)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    
    # Remove y-axis ticks on the main plot to prevent overlap with the left chain bar
    ax.set_yticks([])

    # New! Overlay 1D Track Data if provided
    if track_data:
        # !!!!!!!!!
        track_vals = _get_aligned_track_data(track_data, chain_labels)
        if track_vals is not None:
            # Normalize track values for plotting
            valid_vals = track_vals[~np.isnan(track_vals)]
            if len(valid_vals) > 0:
                vmin, vmax = np.min(valid_vals), np.max(valid_vals)
                norm_vals = (track_vals - vmin) / (vmax - vmin + 1e-8)
                
                # Plot on top axis
                for i in range(N):
                    if not np.isnan(norm_vals[i]):
                        if track_data.get("track_type", "line") == "line":
                            ax_top.plot(i, 1.5, marker='o', color=track_data.get("color", "orange"), markersize=5, alpha=norm_vals[i])
                        elif track_data.get("track_type") == "bar":
                            ax_top.bar(i, 0.5, color=track_data.get("color", "orange"), alpha=norm_vals[i])
                
                # Plot on left axis
                for i in range(N):
                    if not np.isnan(norm_vals[i]):
                        if track_data.get("track_type", "line") == "line":
                            ax_left.plot(1.5, i, marker='o', color=track_data.get("color", "orange"), markersize=5, alpha=norm_vals[i])
                        elif track_data.get("track_type") == "bar":
                            ax_left.barh(i, 0.5, color=track_data.get("color", "orange"), alpha=norm_vals[i])



    # Add Chain Boundaries and Labels
    for cid, s, e in chain_blocks:  
        # Draw boundary lines
        if s != 0:
            sep = s - 0.5
            ax.axvline(sep, color='black', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(sep, color='black', linestyle='--', linewidth=1, alpha=0.7)
            # White lines on the chain bars
            ax_top.axvline(sep, color="w", linewidth=1)
            ax_left.axhline(sep, color="w", linewidth=1)
            
        # Add labels 
        center = (s + e) / 2.0 
        ax_top.text(center, 0, str(cid), ha="center", va="center", 
                    fontsize=14, weight="bold", color="#222222") 
        ax_left.text(0, center, str(cid), ha="center", va="center", 
                     fontsize=14, weight="bold", color="#222222", rotation=90)

    # Set titles and labels
    ax_top.set_title(f"Contact Map: {job_name}", fontsize=14, pad=10) 
    ax.set_xlabel("Token Index (Protein: CA, DNA/RNA: C1', Ligand: Atom1)", fontsize=12) 
    ax.set_ylabel("Token Index", fontsize=12) 

    # 4. Save Output 
    plt.savefig(f"{out_path}/{job_name}_contact_map.pdf", bbox_inches='tight') 
    # also save as png, dpi 300 
    plt.savefig(f"{out_path}/{job_name}_contact_map.png", bbox_inches='tight', dpi=300) 
    plt.close(fig) 


===================================================================================================================================================

# 2, 一个更加成熟的版本, 加上了多维track堆叠的展示
import os
import numpy as np
from typing import Optional, List, Union, Dict, Any
from Bio.PDB import MMCIFParser, Polypeptide
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable 
# fail at multi-track plotting, that's why we import gridspec below, but can be imported again to draw Scale bar

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable 
from alphafold3_seqvis_toolkit.utils.track_utils import parse_bed_to_track_data

def contact_map_vis(
    mmcif_file: str,
    chains: Optional[Union[str, List[str]]] = None,
    out_path: Optional[str] = None,
    cmap: str = "RdBu", # Set RdBu so Red is close (contact), Blue is far, or coolwarm_r
    track_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None # list supported
):
    """
    Description
    -----------
    Generate a contact map (distance matrix) for a single structure from an mmCIF file.
    Uses representative atoms: CA for Protein, C1' for Nucleic Acids, First Atom for others.
    
    Args
    -----
        mmcif_file (str): Path to the mmcif file.
        chains (str | List[str], optional): Chain ID(s) to include. 
                                            e.g., "A" or ["A", "B"]. 
                                            If None, all chains are included.
        out_path (str, optional): Path to save the output plot (e.g., "/data2").
        vmax (float): Maximum distance (Angstrom) for colorbar scaling. Default is 95th percentile of distances.
        cmap (str): Colormap to use. Default is "RdBu_r" (Red=Close, Blue=Far).
        track_data (Dict[str, Any], optional): Additional data to overlay on the contact map.
        track_data (dict/list, optional): 1D feature track data.  
            Format:
            [{
                "track_name": "IDR Score",
                "data": {"A": [0.1, ...], "B": [0.5, ...]}, # Lists must match residue count
                "color": "orange",
                "ylim": (0, 1),
                "track_type": "line" # or 'bar'
            }]
    
    Notes
    ------
    - 1, We currently support one AND list additional track_data overlay.
    - 2, The 1D track data should be prepared externally in some kind certain fomrat, see utils/track_utils.py for details.
    - 3, When not providing 1D track data, we will not plot any overlays.
    """

    job_name = os.path.splitext(os.path.basename(mmcif_file))[0].split("_")[1] 
    def _load_representative_atoms(mmcif_path, target_chains=None):
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('struct', mmcif_path)
        # AF3 usually has model 0
        model = structure[0]

        coords = []
        chain_labels = [] 
        res_types = [] # Track type for info              
        
        # Normalize target_chains
        if isinstance(target_chains, str):
            target_chains = {target_chains}
        elif isinstance(target_chains, (list, tuple)):
            target_chains = set(target_chains)
        
        found_chains = set()

        for chain in model:
            if target_chains is not None and chain.id not in target_chains:
                continue
            
            for res in chain:
                # Skip water 
                if res.id[0] == "W": continue   
                # Note that HETATM residues are included here, cause Zn2+ is HETATM, so we don't skip them
                # but if we want to skip other HETATMs, we can add more conditions here.
                # like: if res.id[0].startswith("H_"): continue
 
                rep_atom = None      
                rtype = "Unknown"

                # 1. Protein -> CA
                if Polypeptide.is_aa(res, standard=True):
                    if 'CA' in res:
                        rep_atom = res['CA']
                        rtype = "Protein"
                
                # 2. Nucleic Acid -> C1' (Distinguish DNA/RNA)
                elif "C1'" in res:
                    rep_atom = res["C1'"]
                    rname = res.resname.strip()
                    if rname in ['DA', 'DT', 'DG', 'DC']:
                        rtype = "DNA"
                    elif rname in ['A', 'U', 'G', 'C']:
                        rtype = "RNA"
                    else:
                        rtype = "Nucleic" # Fallback for modified bases
                
                # 3. Others (Ligands/Ions) -> First Atom
                else:
                    atoms = list(res.get_atoms())
                    if len(atoms) > 0:
                        rep_atom = atoms[0]
                        rtype = "Ligand"

                if rep_atom:
                    coords.append(rep_atom.get_coord())
                    chain_labels.append(chain.id)
                    res_types.append(rtype)
                    found_chains.add(chain.id)

        if not coords:
            msg = f"No valid atoms found in {mmcif_path}."
            if target_chains: msg += f" (Searched for chains: {target_chains})"
            raise ValueError(msg)

        return np.array(coords, dtype=np.float32), chain_labels, sorted(list(found_chains))


    # We need 1D track data support here 
    def _get_aligned_track_data(track_cfg, chain_labels_list):
        """
        Description
        ------------
        Align 1D track data to the residues in the structure.
        Aligns track data (sparse dict) to the flat list of residues in the structure.
        
        Args
        ----
            track_cfg (dict): Configuration for the track data.
            chain_labels_list (list): List of chain IDs corresponding to each residue (rep atom).
        
        Notes
        -----
        - 1, The track_cfg is 0-based index !
        """
        
        # check first
        if not track_cfg or "track_data" not in track_cfg:
            return None
        
        full_data = []
        # our data
        raw_data = track_cfg["track_data"]

        # we traverse all residues in the structure
        current_chain = None
        res_counter = 0

        # Align data, we calculate residue index per chain
        for cid in chain_labels_list:
            if cid != current_chain:
                current_chain = cid
                res_counter = 0 # Note that track_cfg is 0-based index!
            else:
                res_counter += 1

            # default value is nan
            val = np.nan

            # check the raw_data
            if cid in raw_data:
                chain_data = raw_data[cid]
                # get bed value, default is nan
                val = chain_data.get(res_counter, np.nan)

            # now we get the value for this residue
            full_data.append(val)
        # full_data now contains aligned values for all residues/rep atoms in the structure, whether the chain is present in track_data or not
        # that means, full_data is bed value aligned to chains in config_track, and nan for missing data
        # value column in track data can be numerical or categorical (str), so we just keep it 
        return np.asarray(full_data, dtype=object)



    # 1. Load Data
    coords, chain_labels, loaded_chains = _load_representative_atoms(mmcif_file, chains)
    N = coords.shape[0] # number of tokens/representative atoms/residues
    print(f"Loaded {N} tokens from chains: {loaded_chains}")

    # 2. Compute Distance Matrix
    # shape (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # 3. Prepare Tracks
    track_list = []
    if isinstance(track_data, dict): 
        track_list = [track_data]
    elif isinstance(track_data, list): 
        track_list = track_data

    n_tracks = len(track_list)

    # 4. Setup Figure with Gridspec
    # Layout: [Top Tracks...] + [Chain Bar] + [Heatmap]
    # We need n_tracks + 2 rows, and n_tracks + 3 columns (Left Tracks + Chain + Heatmap + Cbar)

    # Ratios
    track_ratio = 0.8  # Track height relative to others
    chain_ratio = 0.5  # Chain bar is thin
    main_ratio = 10.0  # Heatmap is large
    cbar_ratio = 0.4   # Colorbar width

    # Rows: Top Tracks (N) -> Chain Bar (1) -> Heatmap (1)
    height_ratios = [track_ratio] * n_tracks + [chain_ratio, main_ratio]
    
    # Cols: Left Tracks (N) -> Chain Bar (1) -> Heatmap (1) -> Colorbar (1)
    width_ratios = [track_ratio] * n_tracks + [chain_ratio, main_ratio, cbar_ratio]
    
    fig = plt.figure(figsize=(10 + n_tracks, 10 + n_tracks)) # Dynamic size
    gs = gridspec.GridSpec(
        nrows=n_tracks + 2, 
        ncols=n_tracks + 3, 
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        wspace=0.02, hspace=0.02 # Tight gap
    )

    # Indices for the Main Heatmap
    main_row_idx = n_tracks + 1
    main_col_idx = n_tracks + 1
    
    # --- Plot Main Heatmap ---
    ax_main = fig.add_subplot(gs[main_row_idx, main_col_idx])
    im = ax_main.imshow(
        dist_matrix,
        cmap=cmap,
        origin="upper",
        vmin=0,
        interpolation="nearest",
        aspect='auto' # Important for GridSpec
    )
    ax_main.set_yticks([]) # Hide Y ticks
    ax_main.set_xlabel("Token Index", fontsize=12)
    
    # --- Plot Colorbar ---
    ax_cbar = fig.add_subplot(gs[main_row_idx, main_col_idx + 1])
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Distance (Å)", fontsize=12)

    # --- Prepare Chain Blocks ---
    chain_blocks = []
    start = 0
    for i in range(1, N + 1):
        if i == N or chain_labels[i] != chain_labels[start]:
            chain_blocks.append((chain_labels[start], start, i - 1))
            start = i
    
    unique_chains = sorted(list(set(chain_labels)))
    chain_to_int = {cid: i for i, cid in enumerate(unique_chains)}
    chain_row = np.array([chain_to_int[c] for c in chain_labels]).reshape(1, -1)
    
    if len(unique_chains) <= 20: cmap_chains = plt.get_cmap("tab20", len(unique_chains))
    else: cmap_chains = ListedColormap(plt.get_cmap("gist_rainbow")(np.linspace(0, 1, len(unique_chains))))

    # --- Plot Chain Bars ---
    # Top Chain Bar
    ax_chain_top = fig.add_subplot(gs[main_row_idx - 1, main_col_idx], sharex=ax_main)
    ax_chain_top.imshow(chain_row, cmap=cmap_chains, aspect="auto")
    ax_chain_top.set_yticks([])
    ax_chain_top.tick_params(labelbottom=False) # Hide x labels
    
    # Left Chain Bar
    ax_chain_left = fig.add_subplot(gs[main_row_idx, main_col_idx - 1], sharey=ax_main)
    ax_chain_left.imshow(chain_row.T, cmap=cmap_chains, aspect="auto")
    ax_chain_left.set_xticks([])
    ax_chain_left.tick_params(labelleft=False)

    # --- Plot 1D Tracks ---
    # Loop from inner (closest to chain bar) to outer
    for i, track_cfg in enumerate(track_list):
        # Calculate Grid Indices
        # Top tracks stack upwards: index = (main_row - 1) - 1 - i
        row_idx = (main_row_idx - 1) - 1 - i
        col_idx = main_col_idx
        
        # Left tracks stack leftwards: index = (main_col - 1) - 1 - i
        row_idx_l = main_row_idx
        col_idx_l = (main_col_idx - 1) - 1 - i
        
        # Create Axes
        ax_t_top = fig.add_subplot(gs[row_idx, col_idx], sharex=ax_main)
        ax_t_left = fig.add_subplot(gs[row_idx_l, col_idx_l], sharey=ax_main)
        
        # Get Data
        track_vals = _get_aligned_track_data(track_cfg, chain_labels)
        if track_vals is None: continue
            
        t_color = track_cfg.get("color", "orange")
        t_type = track_cfg.get("track_type", "line")
        t_name = track_cfg.get("track_name", "")
        
        x_indices = np.arange(N)
        
        # --- Plotting Logic ---
        if t_type == "categorical":
            # Use imshow for categorical data (cleaner than bar)
            # 1. Map categories to integers
            # Handle NaNs by assigning them to -1 or a specific index
            str_vals = [str(v) for v in track_vals]
            # Get color map
            color_map = t_color if isinstance(t_color, dict) else {}
            
            # Create a list of colors for the colormap
            # We need to ensure the integer mapping matches the colormap order
            unique_cats = sorted(list(set([v for v in str_vals if v != "nan"])))
            cat_to_int = {cat: i for i, cat in enumerate(unique_cats)}
            
            # Create integer array for imshow
            int_row = np.full((1, N), -1) # -1 for background/nan
            for idx, v in enumerate(str_vals):
                if v in cat_to_int:
                    int_row[0, idx] = cat_to_int[v]
            
            # Create Colormap
            # Colors must match the integer indices
            colors_list = [color_map.get(cat, "#808080") for cat in unique_cats]
            if not colors_list: colors_list = ["#FFFFFF"] # Fallback
            custom_cmap = ListedColormap(colors_list)
            
            # Plot Top
            # We need to mask -1 values to be transparent or white
            # Or just set background color of axes
            ax_t_top.imshow(int_row, cmap=custom_cmap, aspect="auto", interpolation="nearest")
            
            # Plot Left (Transpose)
            ax_t_left.imshow(int_row.T, cmap=custom_cmap, aspect="auto", interpolation="nearest")
            
        else:
            # Numerical: Plot Line with Fill (Chi-Score style / Orc1-IDR style)
            try:
                vals = track_vals.astype(float)
            except:
                vals = np.full(N, np.nan)
            
            # Top Plot
            ax_t_top.plot(x_indices, vals, color=t_color, linewidth=1)
            # Fill area (Orc1-IDR style)
            valid_mask = ~np.isnan(vals)
            if np.any(valid_mask):
                min_val = np.nanmin(vals)
                ax_t_top.fill_between(x_indices, vals, min_val, color=t_color, alpha=0.4)
            
            # Left Plot (Rotated)
            ax_t_left.plot(vals, x_indices, color=t_color, linewidth=1)
            if np.any(valid_mask):
                ax_t_left.fill_betweenx(x_indices, vals, min_val, color=t_color, alpha=0.4)
            
            # Limits
            if np.any(valid_mask):
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                margin = (vmax - vmin) * 0.1
                ax_t_top.set_ylim(vmin - margin, vmax + margin)
                ax_t_left.set_xlim(vmin - margin, vmax + margin) # Note xlim for left plot

        # --- Styling ---
        # Hide ticks
        ax_t_top.set_xticks([]); ax_t_top.set_yticks([])
        ax_t_left.set_xticks([]); ax_t_left.set_yticks([])
        
        # Add Label
        ax_t_top.set_ylabel(t_name, fontsize=9, rotation=0, ha="right", va="center")
        
        # Add Chain Boundaries
        for cid, s, e in chain_blocks:
            if s != 0:
                sep = s - 0.5
                ax_t_top.axvline(sep, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
                ax_t_left.axhline(sep, color='black', linestyle='--', linewidth=0.5, alpha=0.3)

    # --- Final Touches ---
    # Add Chain Boundaries to Main Heatmap
    for cid, s, e in chain_blocks:
        if s != 0:
            sep = s - 0.5
            ax_main.axvline(sep, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_main.axhline(sep, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Add Labels to Chain Bars
        center = (s + e) / 2.0
        ax_chain_top.text(center, 0, str(cid), ha="center", va="center", fontsize=10, weight="bold")
        ax_chain_left.text(0, center, str(cid), ha="center", va="center", fontsize=10, weight="bold", rotation=90)

    # Title (Set on the top-most track axes or figure)
    fig.suptitle(f"Contact Map: {job_name}", fontsize=16, y=0.98)

    plt.savefig(f"{out_path}/{job_name}_contact_map.pdf", bbox_inches='tight')
    plt.savefig(f"{out_path}/{job_name}_contact_map.png", bbox_inches='tight', dpi=300)
    plt.close(fig)


======================================================================================================================================================、

# 3, 稍微正常点的一个更新版本
# 残基互作热图+堆叠track+分类图bar未覆盖区域渲染为透明, 但是数值曲线图没有标度, 热图也没有token标度
import os
import re
import numpy as np
from typing import Optional, List, Union, Dict, Any
from Bio.PDB import MMCIFParser, Polypeptide
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable 
# fail at multi-track plotting, that's why we import gridspec below, but can be imported again to draw Scale bar

import matplotlib.gridspec as gridspec
from alphafold3_seqvis_toolkit.utils.track_utils import parse_bed_to_track_data

def contact_map_vis(
    mmcif_file: str,
    chains: Optional[Union[str, List[str]]] = None,
    out_path: Optional[str] = None,
    cmap: str = "RdBu", # Set RdBu so Red is close (contact), Blue is far, or coolwarm_r
    track_bed_file: str = None,
    color_config: Union[str, Dict[str, Any]] = "tab10"
):
    """
    Description
    -----------
    Generate a contact map (distance matrix) for a single structure from an mmCIF file.
    Uses representative atoms: CA for Protein, C1' for Nucleic Acids, First Atom for others.
    
    Args
    -----
        mmcif_file (str): Path to the mmcif file.
        chains (str | List[str], optional): Chain ID(s) to include. 
                                            e.g., "A" or ["A", "B"]. 
                                            If None, all chains are included.
        out_path (str, optional): Path to save the output plot (e.g., "/data2").
        vmax (float): Maximum distance (Angstrom) for colorbar scaling. Default is 95th percentile of distances.
        cmap (str): Colormap to use. Default is "RdBu_r" (Red=Close, Blue=Far).
        track_bed_file (str, optional): Path to the BED-like file containing 1D track data for overlay.see utils/track_utils.py for details.
        color_config (Union[str, Dict[str, Any]]): Color configuration for the 1D tracks.
            - String: A colormap name (e.g. "tab10") or a single color (e.g. "orange").
            - Dict: {TrackName: ColorConfig}.
              e.g. {"IDR": "red", "Domain": {"DomainA": "blue", "DomainB": "green"}}
              or {"IDR": "red", "Domain": "tab10"}
    
    Notes
    ------
    - 1, We currently support one AND list additional track_data overlay.
    - 2, The 1D track data should be prepared externally in some kind certain fomrat, see utils/track_utils.py for details.
    - 3, When not providing 1D track data, we will not plot any overlays.
    - 4, Track data is parsed using parse_bed_to_track_data function from track_utils.py, which needs parameter color_config and track_bed_file.
    The logic is simple: track_bed_file + color_config -> parse_bed_to_track_data() -> track_data for plotting.
    
    parse_bed_to_track_data() will return something like below:

    track_data (Dict[str, Any], optional): Additional data (1D feature track data) to overlay on the contact map.
            Format:
            [{
                "track_name": "IDR",
                "track_type": "categiorical" or "numerical",
                "color": "red" or {"A": "red", "B": "blue"} or "tab10",
                "track_data": {"A": [0.1, ...], "B": [0.5, ...]}, # Lists must match residue count
            }]
    """

    # first, we prepare output path
    job_name = re.match(r'fold_(.*)_model_\d+\.cif', os.path.basename(mmcif_file)).group(1) 
    
    # for computing contact map, we need to load the structure and extract representative atoms
    def _load_representative_atoms(mmcif_path, target_chains=None):
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('struct', mmcif_path)
        # AF3 usually has model 0
        model = structure[0]

        coords = []
        chain_labels = [] 
        res_types = [] # Track type for info              
        
        # Normalize target_chains
        if isinstance(target_chains, str):
            target_chains = {target_chains}
        elif isinstance(target_chains, (list, tuple)):
            target_chains = set(target_chains)
        
        found_chains = set()

        for chain in model:
            if target_chains is not None and chain.id not in target_chains:
                continue
            
            for res in chain:
                # Skip water 
                if res.id[0] == "W": continue   
                # Note that HETATM residues are included here, cause Zn2+ is HETATM, so we don't skip them
                # but if we want to skip other HETATMs, we can add more conditions here.
                # like: if res.id[0].startswith("H_"): continue
 
                rep_atom = None      
                rtype = "Unknown"

                # 1. Protein -> CA
                if Polypeptide.is_aa(res, standard=True):
                    if 'CA' in res:
                        rep_atom = res['CA']
                        rtype = "Protein"
                
                # 2. Nucleic Acid -> C1' (Distinguish DNA/RNA)
                elif "C1'" in res:
                    rep_atom = res["C1'"]
                    rname = res.resname.strip()
                    if rname in ['DA', 'DT', 'DG', 'DC']:
                        rtype = "DNA"
                    elif rname in ['A', 'U', 'G', 'C']:
                        rtype = "RNA"
                    else:
                        rtype = "Nucleic" # Fallback for modified bases
                
                # 3. Others (Ligands/Ions) -> First Atom
                else:
                    atoms = list(res.get_atoms())
                    if len(atoms) > 0:
                        rep_atom = atoms[0]
                        rtype = "Ligand"

                if rep_atom:
                    coords.append(rep_atom.get_coord())
                    chain_labels.append(chain.id)
                    res_types.append(rtype)
                    found_chains.add(chain.id)

        if not coords:
            msg = f"No valid atoms found in {mmcif_path}."
            if target_chains: msg += f" (Searched for chains: {target_chains})"
            raise ValueError(msg)

        return np.array(coords, dtype=np.float32), chain_labels, sorted(list(found_chains))


    # We need 1D track data support here 
    def _get_aligned_track_data(track_cfg, chain_labels_list):
        """
        Description
        ------------
        Align 1D track data to the residues in the structure.
        Aligns track data (sparse dict) to the flat list of residues in the structure.
        
        Args
        ----
            track_cfg (dict): Configuration for the track data.
            chain_labels_list (list): List of chain IDs corresponding to each residue (rep atom).
        
        Notes
        -----
        - 1, The track_cfg is 0-based index !
        """
        
        # check first
        if not track_cfg or "track_data" not in track_cfg:
            return None
        
        full_data = []
        # our data
        raw_data = track_cfg["track_data"]

        # we traverse all residues in the structure
        current_chain = None
        res_counter = 0

        # Align data, we calculate residue index per chain
        for cid in chain_labels_list:
            if cid != current_chain:
                current_chain = cid
                res_counter = 0 # Note that track_cfg is 0-based index!
            else:
                res_counter += 1

            # default value is nan
            val = np.nan

            # check the raw_data
            if cid in raw_data:
                chain_data = raw_data[cid]
                # get bed value, default is nan
                val = chain_data.get(res_counter, np.nan)

            # now we get the value for this residue
            full_data.append(val)
        # full_data now contains aligned values for all residues/rep atoms in the structure, whether the chain is present in track_data or not
        # that means, full_data is bed value aligned to chains in config_track, and nan for missing data
        # value column in track data can be numerical or categorical (str), so we just keep it 
        return np.asarray(full_data, dtype=object)



    # 1. Load Data
    coords, chain_labels, loaded_chains = _load_representative_atoms(mmcif_file, chains)
    N = coords.shape[0] # number of tokens/representative atoms/residues
    print(f"Loaded {N} tokens from chains: {loaded_chains}")

    # 2. Compute Distance Matrix
    # shape (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # 3. Prepare Tracks
    track_data = parse_bed_to_track_data(
        bed_file = track_bed_file,
        color = color_config)
    track_list = []
    if isinstance(track_data, dict): 
        track_list = [track_data]
    # see in utils/track_utils.py, parse_bed_to_track_data() will return a list normally
    elif isinstance(track_data, list): 
        track_list = track_data

    n_tracks = len(track_list)

    # 4. Setup Figure with Gridspec
    # Layout: [Top Tracks...] + [Chain Bar] + [Heatmap]
    # We need n_tracks + 2 rows, and n_tracks + 3 columns (Left Tracks + Chain + Heatmap + Cbar)

    # Ratios
    track_ratio = 0.8  # Track height relative to others
    chain_ratio = 0.5  # Chain bar is thin
    main_ratio = 10.0  # Heatmap is large
    cbar_ratio = 0.4   # Colorbar width

    # Rows: Top Tracks (N) -> Chain Bar (1) -> Heatmap (1)
    height_ratios = [track_ratio] * n_tracks + [chain_ratio, main_ratio]
    
    # Cols: Left Tracks (N) -> Chain Bar (1) -> Heatmap (1) -> Colorbar (1)
    width_ratios = [track_ratio] * n_tracks + [chain_ratio, main_ratio, cbar_ratio]
    
    # Dynamic size according to number of tracks
    fig = plt.figure(figsize=(15 + n_tracks, 15 + n_tracks)) 
    gs = gridspec.GridSpec(
        nrows=n_tracks + 2, # Rows: Top Tracks (N) -> Chain Bar (1) -> Heatmap (1)
        ncols=n_tracks + 3, # Cols: Left Tracks (N) -> Chain Bar (1) -> Heatmap (1) -> Colorbar (1)
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        wspace=0.02, hspace=0.02 # Tight gap
    )

    # Indices for the Main Heatmap
    # cause we have n_tracks top and left, then 1 chain bar, so main heatmap is at (n_tracks+1, n_tracks+1)
    main_row_idx = n_tracks + 1
    main_col_idx = n_tracks + 1
    
    # --- Plot Main Heatmap ---
    ax_main = fig.add_subplot(gs[main_row_idx, main_col_idx])
    im = ax_main.imshow(
        dist_matrix,
        cmap=cmap,
        origin="upper",
        vmin=0,
        interpolation="nearest",
        aspect='auto' # Important for GridSpec
    )
    ax_main.set_yticks([]) # Hide Y ticks
    ax_main.set_xlabel("Token Index", fontsize=12)
    
    # --- Plot Colorbar ---
    ax_cbar = fig.add_subplot(gs[main_row_idx, main_col_idx + 1])
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Distance (Å)", fontsize=12)

    # --- Prepare Chain Blocks ---
    chain_blocks = []
    start = 0
    for i in range(1, N + 1):
        if i == N or chain_labels[i] != chain_labels[start]:
            chain_blocks.append((chain_labels[start], start, i - 1))
            start = i
    
    unique_chains = sorted(list(set(chain_labels)))
    chain_to_int = {cid: i for i, cid in enumerate(unique_chains)}
    chain_row = np.array([chain_to_int[c] for c in chain_labels]).reshape(1, -1)
    
    if len(unique_chains) <= 20: cmap_chains = plt.get_cmap("tab20", len(unique_chains))
    else: cmap_chains = ListedColormap(plt.get_cmap("gist_rainbow")(np.linspace(0, 1, len(unique_chains))))

    # --- Plot Chain Bars ---
    # Top Chain Bar
    ax_chain_top = fig.add_subplot(gs[main_row_idx - 1, main_col_idx], sharex=ax_main)
    ax_chain_top.imshow(chain_row, cmap=cmap_chains, aspect="auto")
    ax_chain_top.set_yticks([])
    ax_chain_top.tick_params(labelbottom=False) # Hide x labels
    
    # Left Chain Bar
    ax_chain_left = fig.add_subplot(gs[main_row_idx, main_col_idx - 1], sharey=ax_main)
    ax_chain_left.imshow(chain_row.T, cmap=cmap_chains, aspect="auto")
    ax_chain_left.set_xticks([])
    ax_chain_left.tick_params(labelleft=False)

    # --- Plot 1D Tracks ---
    # Loop from inner (closest to chain bar) to outer
    for i, track_cfg in enumerate(track_list):
        # Calculate Grid Indices
        # Top tracks stack upwards: index = (main_row - 1) - 1 - i
        row_idx = (main_row_idx - 1) - 1 - i
        col_idx = main_col_idx
        
        # Left tracks stack leftwards: index = (main_col - 1) - 1 - i
        row_idx_l = main_row_idx
        col_idx_l = (main_col_idx - 1) - 1 - i
        
        # Create Axes
        # top track axes, share x with main heatmap
        ax_t_top = fig.add_subplot(gs[row_idx, col_idx], sharex=ax_main)
        # left track axes, share y with main heatmap
        ax_t_left = fig.add_subplot(gs[row_idx_l, col_idx_l], sharey=ax_main)
        
        # Get Aligned Data
        # align track data to the residue list
        track_vals = _get_aligned_track_data(track_cfg, chain_labels)
        if track_vals is None: 
            continue
            
        # Get Track Config
        # track color/type/name, defult value is useless here, so we just set something
        t_color = track_cfg.get("color", "tab10c")
        t_type = track_cfg.get("track_type", "categorical") # categorical or numerical
        t_name = track_cfg.get("track_name", "")
        
        x_indices = np.arange(N)
        
        # --- Plotting Logic ---
        if t_type == "categorical":
            # Use imshow for categorical data (cleaner than bar)
            # 1. Map categories to integers
            # Handle NaNs by assigning them to -1 or a specific index
            str_vals = [str(v) for v in track_vals]
            # Get color map
            color_map = t_color if isinstance(t_color, dict) else {}
            
            # Create a list of colors for the colormap
            # We need to ensure the integer mapping matches the colormap order
            unique_cats = sorted(list(set([v for v in str_vals if v != "nan"])))
            cat_to_int = {cat: i for i, cat in enumerate(unique_cats)}
            
            # Create integer array for imshow
            int_row = np.full((1, N), -1) # -1 for background/nan
            for idx, v in enumerate(str_vals):
                if v in cat_to_int:
                    int_row[0, idx] = cat_to_int[v]
            
            # Create Colormap
            # Colors must match the integer indices
            colors_list = [color_map.get(cat, "#808080") for cat in unique_cats]
            if not colors_list: colors_list = ["#FFFFFF"] # Fallback
            custom_cmap = ListedColormap(colors_list)
            
            # Mask the background (-1) to make it transparent
            masked_row = np.ma.masked_where(int_row == -1, int_row)

            # Plot Top
            # We need to mask -1 values to be transparent or white
            if len(unique_cats) > 0:
                # Determine vmin/vmax to ensure correct color mapping
                vmin, vmax = 0, len(unique_cats) - 1
                if vmin == vmax: # Single category case
                    vmin -= 0.5
                    vmax += 0.5

                ax_t_top.imshow(masked_row, cmap=custom_cmap, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
                
                # Plot Left (Transpose)
                ax_t_left.imshow(masked_row.T, cmap=custom_cmap, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
            
        else:
            # Numerical: Plot Line without Fill (there is no need to fill, cause we do not know the baseline)
            try:
                # Convert to float, invalid parsing will be nan
                vals = track_vals.astype(float)
            except:
                vals = np.full(N, np.nan)
            
            # ---- Top Plot (line) -------
            ax_t_top.plot(x_indices, vals, color=t_color, linewidth=1)
            
            # Get valid mask
            valid_mask = ~np.isnan(vals)

            # ⚠️ Deprecated !
            # Fill area under curve to min value
            '''
            if np.any(valid_mask):
                min_val = np.nanmin(vals)
                ax_t_top.fill_between(x_indices, vals, min_val, color=t_color, alpha=0.4)
            '''
            
            # Left Plot (Rotated)
            ax_t_left.plot(vals, x_indices, color=t_color, linewidth=1)
            
            # ⚠️ Deprecated also !
            # Fill area to min value
            '''
            if np.any(valid_mask):
                ax_t_left.fill_betweenx(x_indices, vals, min_val, color=t_color, alpha=0.4)
            '''
            
            # Limits
            if np.any(valid_mask):
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                # ⚠️ Add some margin, 10% of range
                # if the output plot is not good, we can adjust this margin ratio, or just skip margin for original data range
                margin = (vmax - vmin) * 0.1
                # So that line is not at the edge, the actual data range is larger than vmin/vmax
                # the curve is vmin/vmax, but the track axes limit is extended, so line is not at the edge
                ax_t_top.set_ylim(vmin - margin, vmax + margin)
                ax_t_left.set_xlim(vmin - margin, vmax + margin) # Note xlim for left plot

        # --- Styling ---
        # Hide ticks
        ax_t_top.set_xticks([]); ax_t_top.set_yticks([])
        ax_t_left.set_xticks([]); ax_t_left.set_yticks([])
        
        # Add Label
        ax_t_top.set_ylabel(t_name, fontsize=9, rotation=0, ha="right", va="center")
        
        # Add Chain Boundaries
        for cid, s, e in chain_blocks:
            if s != 0:
                sep = s - 0.5
                ax_t_top.axvline(sep, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
                ax_t_left.axhline(sep, color='black', linestyle='--', linewidth=0.5, alpha=0.3)

    # --- Final Touches ---
    # Add Chain Boundaries to Main Heatmap
    for cid, s, e in chain_blocks:
        if s != 0:
            sep = s - 0.5
            ax_main.axvline(sep, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_main.axhline(sep, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Add Labels to Chain Bars
        center = (s + e) / 2.0
        ax_chain_top.text(center, 0, str(cid), ha="center", va="center", fontsize=10, weight="bold")
        ax_chain_left.text(0, center, str(cid), ha="center", va="center", fontsize=10, weight="bold", rotation=90)

    # Title (Set on the top-most track axes or figure)
    fig.suptitle(f"Contact Map: {job_name}", fontsize=16, y=0.9) # default y=0.98

    plt.savefig(f"{out_path}/{job_name}_contact_map.pdf", bbox_inches='tight')
    plt.savefig(f"{out_path}/{job_name}_contact_map.png", bbox_inches='tight', dpi=300)
    plt.close(fig)


==========================================================================================================================================
# 4, 在v3的基础上增添了x轴标度，但是是整体结构的标度，没有区分每一个chain内部自己的坐标
import os
import re
import numpy as np
from typing import Optional, List, Union, Dict, Any
from Bio.PDB import MMCIFParser, Polypeptide
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable 
# fail at multi-track plotting, that's why we import gridspec below, but can be imported again to draw Scale bar

import matplotlib.gridspec as gridspec
from alphafold3_seqvis_toolkit.utils.track_utils import parse_bed_to_track_data

def contact_map_vis(
    mmcif_file: str,
    chains: Optional[Union[str, List[str]]] = None,
    out_path: Optional[str] = None,
    cmap: str = "RdBu", # Set RdBu so Red is close (contact), Blue is far, or coolwarm_r
    track_bed_file: str = None,
    color_config: Union[str, Dict[str, Any]] = "tab10"
):
    """
    Description
    -----------
    Generate a contact map (distance matrix) for a single structure from an mmCIF file.
    Uses representative atoms: CA for Protein, C1' for Nucleic Acids, First Atom for others.
    
    Args
    -----
        mmcif_file (str): Path to the mmcif file.
        chains (str | List[str], optional): Chain ID(s) to include. 
                                            e.g., "A" or ["A", "B"]. 
                                            If None, all chains are included.
        out_path (str, optional): Path to save the output plot (e.g., "/data2").
        vmax (float): Maximum distance (Angstrom) for colorbar scaling. Default is 95th percentile of distances.
        cmap (str): Colormap to use. Default is "RdBu_r" (Red=Close, Blue=Far).
        track_bed_file (str, optional): Path to the BED-like file containing 1D track data for overlay.see utils/track_utils.py for details.
        color_config (Union[str, Dict[str, Any]]): Color configuration for the 1D tracks.
            - String: A colormap name (e.g. "tab10") or a single color (e.g. "orange").
            - Dict: {TrackName: ColorConfig}.
              e.g. {"IDR": "red", "Domain": {"DomainA": "blue", "DomainB": "green"}}
              or {"IDR": "red", "Domain": "tab10"}
    
    Notes
    ------
    - 1, We currently support one AND list additional track_data overlay.
    - 2, The 1D track data should be prepared externally in some kind certain fomrat, see utils/track_utils.py for details.
    - 3, When not providing 1D track data, we will not plot any overlays.
    - 4, Track data is parsed using parse_bed_to_track_data function from track_utils.py, which needs parameter color_config and track_bed_file.
    The logic is simple: track_bed_file + color_config -> parse_bed_to_track_data() -> track_data for plotting.
    
    parse_bed_to_track_data() will return something like below:

    track_data (Dict[str, Any], optional): Additional data (1D feature track data) to overlay on the contact map.
            Format:
            [{
                "track_name": "IDR",
                "track_type": "categiorical" or "numerical",
                "color": "red" or {"A": "red", "B": "blue"} or "tab10",
                "track_data": {"A": [0.1, ...], "B": [0.5, ...]}, # Lists must match residue count
            }]
    """

    # first, we prepare output path
    job_name = re.match(r'fold_(.*)_model_\d+\.cif', os.path.basename(mmcif_file)).group(1) 
    
    # for computing contact map, we need to load the structure and extract representative atoms
    def _load_representative_atoms(mmcif_path, target_chains=None):
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('struct', mmcif_path)
        # AF3 usually has model 0
        model = structure[0]

        coords = []
        chain_labels = [] 
        res_types = [] # Track type for info              
        
        # Normalize target_chains
        if isinstance(target_chains, str):
            target_chains = {target_chains}
        elif isinstance(target_chains, (list, tuple)):
            target_chains = set(target_chains)
        
        found_chains = set()

        for chain in model:
            if target_chains is not None and chain.id not in target_chains:
                continue
            
            for res in chain:
                # Skip water 
                if res.id[0] == "W": continue   
                # Note that HETATM residues are included here, cause Zn2+ is HETATM, so we don't skip them
                # but if we want to skip other HETATMs, we can add more conditions here.
                # like: if res.id[0].startswith("H_"): continue
 
                rep_atom = None      
                rtype = "Unknown"

                # 1. Protein -> CA
                if Polypeptide.is_aa(res, standard=True):
                    if 'CA' in res:
                        rep_atom = res['CA']
                        rtype = "Protein"
                
                # 2. Nucleic Acid -> C1' (Distinguish DNA/RNA)
                elif "C1'" in res:
                    rep_atom = res["C1'"]
                    rname = res.resname.strip()
                    if rname in ['DA', 'DT', 'DG', 'DC']:
                        rtype = "DNA"
                    elif rname in ['A', 'U', 'G', 'C']:
                        rtype = "RNA"
                    else:
                        rtype = "Nucleic" # Fallback for modified bases
                
                # 3. Others (Ligands/Ions) -> First Atom
                else:
                    atoms = list(res.get_atoms())
                    if len(atoms) > 0:
                        rep_atom = atoms[0]
                        rtype = "Ligand"

                if rep_atom:
                    coords.append(rep_atom.get_coord())
                    chain_labels.append(chain.id)
                    res_types.append(rtype)
                    found_chains.add(chain.id)

        if not coords:
            msg = f"No valid atoms found in {mmcif_path}."
            if target_chains: msg += f" (Searched for chains: {target_chains})"
            raise ValueError(msg)

        return np.array(coords, dtype=np.float32), chain_labels, sorted(list(found_chains))


    # We need 1D track data support here 
    def _get_aligned_track_data(track_cfg, chain_labels_list):
        """
        Description
        ------------
        Align 1D track data to the residues in the structure.
        Aligns track data (sparse dict) to the flat list of residues in the structure.
        
        Args
        ----
            track_cfg (dict): Configuration for the track data.
            chain_labels_list (list): List of chain IDs corresponding to each residue (rep atom).
        
        Notes
        -----
        - 1, The track_cfg is 0-based index !
        """
        
        # check first
        if not track_cfg or "track_data" not in track_cfg:
            return None
        
        full_data = []
        # our data
        raw_data = track_cfg["track_data"]

        # we traverse all residues in the structure
        current_chain = None
        res_counter = 0

        # Align data, we calculate residue index per chain
        for cid in chain_labels_list:
            if cid != current_chain:
                current_chain = cid
                res_counter = 0 # Note that track_cfg is 0-based index!
            else:
                res_counter += 1

            # default value is nan
            val = np.nan

            # check the raw_data
            if cid in raw_data:
                chain_data = raw_data[cid]
                # get bed value, default is nan
                val = chain_data.get(res_counter, np.nan)

            # now we get the value for this residue
            full_data.append(val)
        # full_data now contains aligned values for all residues/rep atoms in the structure, whether the chain is present in track_data or not
        # that means, full_data is bed value aligned to chains in config_track, and nan for missing data
        # value column in track data can be numerical or categorical (str), so we just keep it 
        return np.asarray(full_data, dtype=object)



    # 1. Load Data
    coords, chain_labels, loaded_chains = _load_representative_atoms(mmcif_file, chains)
    N = coords.shape[0] # number of tokens/representative atoms/residues
    print(f"Loaded {N} tokens from chains: {loaded_chains}")

    # 2. Compute Distance Matrix
    # shape (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # 3. Prepare Tracks
    track_data = parse_bed_to_track_data(
        bed_file = track_bed_file,
        color = color_config)
    track_list = []
    if isinstance(track_data, dict): 
        track_list = [track_data]
    # see in utils/track_utils.py, parse_bed_to_track_data() will return a list normally
    elif isinstance(track_data, list): 
        track_list = track_data

    n_tracks = len(track_list)

    # 4. Setup Figure with Gridspec
    # Layout: [Top Tracks...] + [Chain Bar] + [Heatmap]
    # We need n_tracks + 2 rows, and n_tracks + 3 columns (Left Tracks + Chain + Heatmap + Cbar)

    # Ratios
    track_ratio = 0.8  # Track height relative to others
    chain_ratio = 0.5  # Chain bar is thin
    main_ratio = 10.0  # Heatmap is large
    cbar_ratio = 0.4   # Colorbar width

    # Rows: Top Tracks (N) -> Chain Bar (1) -> Heatmap (1)
    height_ratios = [track_ratio] * n_tracks + [chain_ratio, main_ratio]
    
    # Cols: Left Tracks (N) -> Chain Bar (1) -> Heatmap (1) -> Colorbar (1)
    width_ratios = [track_ratio] * n_tracks + [chain_ratio, main_ratio, cbar_ratio]
    
    # Dynamic size according to number of tracks
    fig = plt.figure(figsize=(15 + n_tracks, 15 + n_tracks)) 
    gs = gridspec.GridSpec(
        nrows=n_tracks + 2, # Rows: Top Tracks (N) -> Chain Bar (1) -> Heatmap (1)
        ncols=n_tracks + 3, # Cols: Left Tracks (N) -> Chain Bar (1) -> Heatmap (1) -> Colorbar (1)
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        wspace=0.02, hspace=0.02 # Tight gap
    )

    # Indices for the Main Heatmap
    # cause we have n_tracks top and left, then 1 chain bar, so main heatmap is at (n_tracks+1, n_tracks+1)
    main_row_idx = n_tracks + 1
    main_col_idx = n_tracks + 1
    
    # --- Plot Main Heatmap ---
    ax_main = fig.add_subplot(gs[main_row_idx, main_col_idx])
    im = ax_main.imshow(
        dist_matrix,
        cmap=cmap,
        origin="upper",
        vmin=0,
        interpolation="nearest",
        aspect='auto' # Important for GridSpec
    )
    ax_main.set_yticks([]) # Hide Y ticks
    ax_main.set_xlabel("Token Index", fontsize=12)
    
    # --- Plot Colorbar ---
    ax_cbar = fig.add_subplot(gs[main_row_idx, main_col_idx + 1])
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Distance (Å)", fontsize=12)

    # --- Prepare Chain Blocks ---
    chain_blocks = []
    start = 0
    for i in range(1, N + 1):
        if i == N or chain_labels[i] != chain_labels[start]:
            chain_blocks.append((chain_labels[start], start, i - 1))
            start = i
    
    unique_chains = sorted(list(set(chain_labels)))
    chain_to_int = {cid: i for i, cid in enumerate(unique_chains)}
    chain_row = np.array([chain_to_int[c] for c in chain_labels]).reshape(1, -1)
    
    if len(unique_chains) <= 20: cmap_chains = plt.get_cmap("tab20", len(unique_chains))
    else: cmap_chains = ListedColormap(plt.get_cmap("gist_rainbow")(np.linspace(0, 1, len(unique_chains))))

    # --- Plot Chain Bars ---
    # Top Chain Bar
    ax_chain_top = fig.add_subplot(gs[main_row_idx - 1, main_col_idx], sharex=ax_main)
    ax_chain_top.imshow(chain_row, cmap=cmap_chains, aspect="auto")
    ax_chain_top.set_yticks([])
    # ⚠️⚠️ax_chain_top.tick_params(labelbottom=False) # Hide x labels
    # FIX: Use tick_params to hide ticks visually instead of removing them, to preserve ax_main ticks
    ax_chain_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False) 
    
    # Left Chain Bar
    ax_chain_left = fig.add_subplot(gs[main_row_idx, main_col_idx - 1], sharey=ax_main)
    ax_chain_left.imshow(chain_row.T, cmap=cmap_chains, aspect="auto")
    ax_chain_left.set_xticks([])
    # ⚠️⚠️ax_chain_left.tick_params(labelleft=False)
    # FIX: Use tick_params to hide ticks visually
    ax_chain_left.tick_params(axis='y', which='both', left=False, labelleft=False)

    # --- Plot 1D Tracks ---
    # Loop from inner (closest to chain bar) to outer
    for i, track_cfg in enumerate(track_list):
        # Calculate Grid Indices
        # Top tracks stack upwards: index = (main_row - 1) - 1 - i
        row_idx = (main_row_idx - 1) - 1 - i
        col_idx = main_col_idx
        
        # Left tracks stack leftwards: index = (main_col - 1) - 1 - i
        row_idx_l = main_row_idx
        col_idx_l = (main_col_idx - 1) - 1 - i
        
        # Create Axes
        # top track axes, share x with main heatmap
        ax_t_top = fig.add_subplot(gs[row_idx, col_idx], sharex=ax_main)
        # left track axes, share y with main heatmap
        ax_t_left = fig.add_subplot(gs[row_idx_l, col_idx_l], sharey=ax_main)
        
        # Get Aligned Data
        # align track data to the residue list
        track_vals = _get_aligned_track_data(track_cfg, chain_labels)
        if track_vals is None: 
            continue
            
        # Get Track Config
        # track color/type/name, defult value is useless here, so we just set something
        t_color = track_cfg.get("color", "tab10c")
        t_type = track_cfg.get("track_type", "categorical") # categorical or numerical
        t_name = track_cfg.get("track_name", "")
        
        x_indices = np.arange(N)
        
        # --- Plotting Logic ---
        if t_type == "categorical":
            # Use imshow for categorical data (cleaner than bar)
            # 1. Map categories to integers
            # Handle NaNs by assigning them to -1 or a specific index
            str_vals = [str(v) for v in track_vals]
            # Get color map
            color_map = t_color if isinstance(t_color, dict) else {}
            
            # Create a list of colors for the colormap
            # We need to ensure the integer mapping matches the colormap order
            unique_cats = sorted(list(set([v for v in str_vals if v != "nan"])))
            cat_to_int = {cat: i for i, cat in enumerate(unique_cats)}
            
            # Create integer array for imshow
            int_row = np.full((1, N), -1) # -1 for background/nan
            for idx, v in enumerate(str_vals):
                if v in cat_to_int:
                    int_row[0, idx] = cat_to_int[v]
            
            # Create Colormap
            # Colors must match the integer indices
            colors_list = [color_map.get(cat, "#808080") for cat in unique_cats]
            if not colors_list: colors_list = ["#FFFFFF"] # Fallback
            custom_cmap = ListedColormap(colors_list)
            
            # Mask the background (-1) to make it transparent
            masked_row = np.ma.masked_where(int_row == -1, int_row)

            # Plot Top
            # We need to mask -1 values to be transparent or white
            if len(unique_cats) > 0:
                # Determine vmin/vmax to ensure correct color mapping
                vmin, vmax = 0, len(unique_cats) - 1
                if vmin == vmax: # Single category case
                    vmin -= 0.5
                    vmax += 0.5

                ax_t_top.imshow(masked_row, cmap=custom_cmap, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
                
                # Plot Left (Transpose)
                ax_t_left.imshow(masked_row.T, cmap=custom_cmap, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
            
        else:
            # Numerical: Plot Line without Fill (there is no need to fill, cause we do not know the baseline)
            try:
                # Convert to float, invalid parsing will be nan
                vals = track_vals.astype(float)
            except:
                vals = np.full(N, np.nan)
            
            # ---- Top Plot (line) -------
            ax_t_top.plot(x_indices, vals, color=t_color, linewidth=1)
            
            # Get valid mask
            valid_mask = ~np.isnan(vals)

            # ⚠️ Deprecated !
            # Fill area under curve to min value
            '''
            if np.any(valid_mask):
                min_val = np.nanmin(vals)
                ax_t_top.fill_between(x_indices, vals, min_val, color=t_color, alpha=0.4)
            '''
            
            # Left Plot (Rotated)
            ax_t_left.plot(vals, x_indices, color=t_color, linewidth=1)
            
            # ⚠️ Deprecated also !
            # Fill area to min value
            '''
            if np.any(valid_mask):
                ax_t_left.fill_betweenx(x_indices, vals, min_val, color=t_color, alpha=0.4)
            '''
            
            # Limits
            if np.any(valid_mask):
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                # ⚠️ Add some margin, 10% of range
                # if the output plot is not good, we can adjust this margin ratio, or just skip margin for original data range
                margin = (vmax - vmin) * 0.1
                # So that line is not at the edge, the actual data range is larger than vmin/vmax
                # the curve is vmin/vmax, but the track axes limit is extended, so line is not at the edge
                ax_t_top.set_ylim(vmin - margin, vmax + margin)
                ax_t_left.set_xlim(vmin - margin, vmax + margin) # Note xlim for left plot

        # --- Styling ---
        # Hide ticks
        # ⚠️⚠️FIX: Do not use set_xticks([]) on shared axes (ax_t_top shares X, ax_t_left shares Y)
        # This prevents wiping out the ticks on the main heatmap
        
        # ax_t_top.set_xticks([]); ax_t_top.set_yticks([])
        # ax_t_left.set_xticks([]); ax_t_left.set_yticks([])
        
        # Add Label
        # ax_t_top.set_ylabel(t_name, fontsize=9, rotation=0, ha="right", va="center")
        # Top Track: Shares X. Hide X ticks/labels visually. Y is independent.
        ax_t_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax_t_top.set_yticks([])
        
        # Left Track: Shares Y. Hide Y ticks/labels visually. X is independent.
        ax_t_left.set_xticks([])
        ax_t_left.tick_params(axis='y', which='both', left=False, labelleft=False)
        
        # Add Label
        ax_t_top.set_ylabel(t_name, fontsize=9, rotation=0, ha="right", va="center")



        # Add Chain Boundaries
        for cid, s, e in chain_blocks:
            if s != 0:
                sep = s - 0.5
                ax_t_top.axvline(sep, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
                ax_t_left.axhline(sep, color='black', linestyle='--', linewidth=0.5, alpha=0.3)

    # --- Final Touches ---
    # Add Chain Boundaries to Main Heatmap
    for cid, s, e in chain_blocks:
        if s != 0:
            sep = s - 0.5
            ax_main.axvline(sep, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_main.axhline(sep, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Add Labels to Chain Bars
        center = (s + e) / 2.0
        ax_chain_top.text(center, 0, str(cid), ha="center", va="center", fontsize=10, weight="bold")
        ax_chain_left.text(0, center, str(cid), ha="center", va="center", fontsize=10, weight="bold", rotation=90)

    # Title (Set on the top-most track axes or figure)
    fig.suptitle(f"Contact Map: {job_name}", fontsize=16, y=0.9) # default y=0.98

    plt.savefig(f"{out_path}/{job_name}_contact_map.pdf", bbox_inches='tight')
    plt.savefig(f"{out_path}/{job_name}_contact_map.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

