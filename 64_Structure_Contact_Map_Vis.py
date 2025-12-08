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
