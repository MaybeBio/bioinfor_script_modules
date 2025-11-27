# Note: 

# Input: 2 mmcif files generated from AlphaFold3 predictions
# - 1, the object compared in 2 mmcif files should be the same protein sequence. For example, both files are predictions of the same protein sequence but under different conditions or with different ligands.

# Output: A 2x2 contact difference map
# - 1, leftupper: Contact map of structure 1, rightupper: Contact map of structure 2, leftlower: Contact difference map (structure 1 - structure 2), rightlower: Contact difference map (structure 2 - structure 1).

# reference: https://biopython.org/docs/1.75/api/Bio.PDB.html


from typing import Optional, Tuple, List, Union
from Bio.PDB import MMCIFParser, Polypeptide
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import os

def contact_map_diff(
        mmcif_file_a: str,
        mmcif_file_b: str,
        region_1: Union[Tuple[int, int], List[int], str],
        region_2: Optional[Union[Tuple[int, int], List[int], str]] = None,  
        region_pairs: Optional[Union[List[Tuple[Tuple[int, int], Tuple[int, int]]], List[str]]] = None,
        vmax : Optional[float] = None,
        vmax_percentile: float = 95.0,
        vdiff: Optional[float] = None,
        vdiff_percentile: float = 95.0,
        include_nonstandard_residue: bool = False,
        return_maxtrix = False,
        out_file: Optional[str] = None,
        cmap_dist = "RdBu", 
        cmap_diff = "seismic"
):
    """
    Description
    --
        Generate a 2x2 contact difference map from two mmcif files.

    Args
        mmcif_file_a (str): Path to the first mmcif file.
        mmcif_file_b (str): Path to the second mmcif file.
        region_1 (Tuple[int, int] | List[int, int]): 0-based Region of interest in the first structure (start, end) or list of residue indices.
            for example, (10, 50) or [10, 11, 12, ..., 50], or [10, 50]
        region_2 (Tuple[int, int] | List[int, int], optional): 0-based Region of interest in the second structure (start, end) or list of residue indices. Defaults to None, which means using region_1.
        regions_pair (List[Tuple[Tuple, Tuple]] | List[str], optional): List of region pairs for batch processing. Each item is a tuple of two regions (region_1, region_2) or a string "start1:end1,start2:end2". Defaults to None.
        vmax (float): Maximum distance value for color bar scaling. If None, it will be determined automatically based on the data. Defaults to None.
        vmax_percentile (float): Percentile value to determine vmax if vmax is None. Defaults to 95.0.
        vdiff (float): Maximum absolute difference value for color bar scaling. If None, it will be determined automatically based on the data. Defaults to None.
        vdiff_percentile (float): Percentile value to determine vdiff if vdiff is

        include_nonstandard_residue (bool): Whether to include non-standard residues like MSE. Defaults to False.
        return_maxtrix (bool): Whether to return the contact matrices of the selected regions and related information. Defaults to False.
        cmap_dist (str): Colormap for distance maps. Defaults to "RdBu".(red for close, blue for far)
        cmap_diff (str): Colormap for difference maps. Defaults to "seismic".(red for positive, blue for negative)

    Returns
        None: Displays a 2x2 contact difference map.

    Notes:
    - 1, region_1 and region_2 are both 0-based residue indices, like CTCF (0,726)
    - 2, current logic only supports like: if only region_1 provided, then we draw region_1 x region_1 submaxtrix;
    if both region_1 and region_2 are provided, then we draw region_1 x region_2 submatrix.
    - 3, if regions_pair is provided, region_1 and region_2 will be ignored, and we will batch process all region pairs in regions_pair.
    Format of regions_pair can be:
        - a, List of tuple of tuples: [ ((start1, end1), (start2, end2)), ... ]
        - b, List of strings: [ "start1:end1,start2:end2", ... ], ["start1-end1,start2-end2"]
    
    
    Todos:
    - 1, support input region format like region1xregion, like 10:50x100:150, so we can directly compare the diamond region between two domains.
    """
    
    # we need get the correct region index first
    # _parse_region for single region
    def _parse_region(r):
        """
        supporting format like (start, end) or [start, end] or "start:end" or "start-end"
        """
        if isinstance(r,(tuple,list)) and len(r) == 2:
            s, e = int(r[0]), int(r[1])
        elif isinstance(r, str) and (":" in r or "-" in r):
            sep = ":" if ":" in r else "-"
            s, e = r.split(sep)
            s, e = int(s.strip()), int(e.strip())
        else:
            raise ValueError(f"Illegal region format: {r}")
        if e < s:
            raise ValueError(f"Illegal region format: {r}, end < start.")
        return (s, e)

    # _parse_region_pairs for batch region pairs
    def _parse_region_pairs(pair_str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        supporting format like "start1:end1,start2:end2" or "start1-end1,start2-end2"
        """
        pair_str = pair_str.strip()
        sep = "," if "," in pair_str else None
        if not sep:
            raise ValueError(f"Illegal region pair format (missing ','): {pair_str}")
        left, right = pair_str.split(sep, 1)
        r1 = _parse_region(left.strip())
        r2 = _parse_region(right.strip())
        return (r1, r2)
    
    # then, we load the result of structure prediction
    # here, we only focus on CA atoms ! ! !
    # And note that each residue should have only one CA atom, so we can directly use residue index to access CA atom
    def _load_ca(mmcif_file, include_nonstandard_residue=False, model_index=0):
        """
        mmcif_file: str, path to the mmcif file
        include_nonstandard_residue: bool, whether to include non-standard residues like MSE
        model_index: int, index of the model to load (default: 0)
        """
        
        # for af3, we generally need only the first model, that is model_0
        parser = MMCIFParser()
        # According to SMCRA hierarchy, we need to go through Structure -> Model -> Chain -> Residue -> Atom
        structure = parser.get_structure('struct', mmcif_file)
        models = list(structure)
        if not models:
            raise ValueError(f"No models found in the mmcif file: {mmcif_file}")
        # Generally, there is only one model in the mmcif file generated by AlphaFold3
        model = models[model_index]

        ca_coords, ca_info = [], []
        for chain in model: 
            for res in chain:
                hetfield, resseq, icode = res.id # or .get_id()
                if hetfield != " ":
                    # that means it's a hetero atom, we skip it
                    continue
                if not Polypeptide.is_aa(res, standard=not include_nonstandard_residue):
                    # that means it's a non-standard residues/amino acids, we also skip it
                    continue
                if 'CA' not in res:
                    # that means there is no CA atom in this residue, we skip it
                    continue
                ca = res['CA']
                # get the coordinate of CA atom
                ca_coords.append(ca.get_coord())
                ca_info.append(
                    {
                        "chain": chain.id, # like "A"
                        "resname": res.resname, # like "ALA"
                        "resseq": resseq, # like position 1(1-based)
                        "icode": (icode if isinstance(icode, str) and icode.strip() else "") # like 'A' for Thr 80 A，Ser 80 B
                    }
                )

        if not ca_coords:
            raise ValueError(f"No CA atoms found in the mmcif file: {mmcif_file}")
        # force convert to np.float32 to save memory
        return np.asarray(ca_coords, dtype=np.float32), ca_info
    
    def _pairwise_dist(ca_coords):
        """
        For a protein structure with N residues(equals N CA atoms), compute the pairwise distance matrix of shape (N, N)
        ca_coords: np.ndarray of shape (N, 3), coordinates of CA atoms
        """

        # genarally we may use double for loop to compute pairwise distance matrix
        # but here we use broadcasting to accelerate the computation
        diff = ca_coords[:, None, :] - ca_coords[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))
    
    # load ca coordinates and info from mmcif files
    ca_coords_a, ca_info_a = _load_ca(mmcif_file_a, include_nonstandard_residue=include_nonstandard_residue)
    ca_coords_b, ca_info_b = _load_ca(mmcif_file_b, include_nonstandard_residue=include_nonstandard_residue)
    # calculate the shape of each protein structure
    # cause we are evaluateing the same protein sequence, so Na should be equal to Nb
    Na, Nb = ca_coords_a.shape[0], ca_coords_b.shape[0]
    if Na != Nb:
        raise ValueError(f"The number of residues in the protein part of two structures are different: {Na} vs {Nb}. Please check whether the two mmcif files are generated from the same protein sequence,\n \
                        cause we are only focusing on the protein regions, so they should match.")
    N = Na  # since Na == Nb

    # parse the region of interest
    pairs: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []
    # if region_pairs is provided, we will prefer it for batch processing, otherwise we use region_1 and region_2
    if region_pairs:
        # format like [((1,2),(3,4)), ...] or ["1:2,3:4", ...]
        # if item is str, we need parse it into tuple of tuples (the first format) first
        if isinstance(region_pairs[0], str):
            pairs = [_parse_region_pairs(p) for p in region_pairs]
        else:
            pairs = [(_parse_region(a), _parse_region(b)) for (a,b) in region_pairs]
            # pairs = region_pairs
    # else, we use region_1 and region_2
    # if only region_1 is provided, we compare region_1 x region_1
    else:
        r1 = _parse_region(region_1)
        r2 = _parse_region(region_2) if region_2 is not None else r1
        pairs = [(r1, r2)]

    # check whether the region is valid
    # Note: we are using 0-based residue index here
    # Note again: we are comparing regions between two structures with the same protein sequence, so N1 = N2 
    for (ra, rb) in pairs:
        for r in (ra, rb):
            if r[0] < 0 or r[1] > N-1:
                raise ValueError(f"Region {r} is out of bounds for protein size {N} (0-based).")

    # calculate pairwise distance matrices for both structures
    dist_a = _pairwise_dist(ca_coords_a)
    dist_b = _pairwise_dist(ca_coords_b)
    diff_ab = dist_a - dist_b
    diff_ba = - diff_ab

    # calculate the color bar limits
    # Here, we want to compare the distance maps at the same scale, so we should unify vmax from the real data of both max values
    # Generally, we may use the max value of both dist maximums as the unified vmax
    # But considering that there may be some extreme values in the distance maps and they maybe noise, we use the 95th percentile value instead
    # For minimum value, we do not set 5th percentile, because the minimum distance should be 0 anyway, it is rational
    # So we set the vmax_percentile parmeter to contorl the color bar limits, default is 95.0, and 100 means using the real max value
    if vmax is None:
        vmax_use = float(max(
            np.nanpercentile(dist_a, vmax_percentile),
            np.nanpercentile(dist_b, vmax_percentile)
        ))
    else:
        vmax_use = float(vmax)

    # the same for vdiff (0 centered)
    if vdiff is None:
        abs_all = np.abs(np.concatenate([diff_ab.ravel(),diff_ba.ravel()]))
        vdiff_use = float(np.nanpercentile(abs_all, vdiff_percentile))
    else:
        vdiff_use = float(vdiff)

    # set 2x2 subplots, better layout with constrained_layout=True
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

    # plot distance map of structure 1, protein marked as A/1
    # A/1
    imA = axes[0,0].imshow(
        dist_a, 
        cmap=cmap_dist,
        vmin=0,
        vmax=vmax_use,
        interpolation="nearest",
        origin="upper"
    )
    axes[0,0].set_title(f"A:{os.path.splitext(os.path.basename(mmcif_file_a))[0].split('_')[1]}")
    
    # plot distance map of structure 2, protein marked as B/2
    # B/2
    imB = axes[0,1].imshow(
        dist_b,
        cmap=cmap_dist,
        vmin=0,
        vmax=vmax_use,
        interpolation="nearest",
        origin="upper"
    )
    axes[0,1].set_title(f"B:{os.path.splitext(os.path.basename(mmcif_file_b))[0].split('_')[1]}")
    

    # plot distance difference map of structure 1 - structure 2, marked as A-B/1-2
    # A-B/1-2

    # Here we use TwoSlopeNorm instead of LinearNorm to ensure that 0 is at the center of the color bar
    # Customize color mapping rules for zero-bounded difference data
    norm = TwoSlopeNorm(vmin=-vdiff_use, vcenter=0, vmax=vdiff_use)
    imAB = axes[1,0].imshow(
        diff_ab,
        cmap=cmap_diff,
        norm=norm,
        interpolation="nearest",
        origin="upper"
    )
    axes[1,0].set_title("A - B (Red = closer, blue = farther)")

    # plot distance difference map of structure 2 - structure 1, marked as B-A/2-1
    # B-A/2-1
    imBA = axes[1,1].imshow(
        diff_ba,
        cmap=cmap_diff,
        norm=norm,
        interpolation="nearest",
        origin="upper"
    )
    axes[1,1].set_title("B - A (Red = closer, blue = farther)")


    # The following annotated function is used to draw the starting and ending points of a region on the axis
    '''
    # annotate the region of interest:
    # '#FFA500'(orange) for region_1, '#800080'(purple) for region_2, avoid using red/blue, which are used in the distance/difference maps
    def _mark_regions(ax, r1, r2=None):
        r1s, r1e = r1
        for pos in (r1s, r1e):
            ax.axhline(pos, color='#FFA500', linestyle='--', linewidth=1.2, zorder=5)
            ax.axvline(pos, color='#FFA500', linestyle='--', linewidth=1.2, zorder=5)
        if r2 is not None:
            r2s, r2e = r2
            for pos in (r2s, r2e):
                ax.axhline(pos, color='#800080', linestyle='--', linewidth=1.2, zorder=5)
                ax.axvline(pos, color='#800080', linestyle='--', linewidth=1.2, zorder=5)
    '''
    # annotate the region of interest using Rectangle patch:
    def _mark_pairs(ax, region_pairs, color="#00FF00"):
        for (ra, rb) in region_pairs:
            a_s, a_e = ra
            b_s, b_e = rb
            ha = a_e - a_s + 1
            hb = b_e - b_s + 1
            # draw ra x rb rectangle (rows=ra, cols=rb)
            ax.add_patch(
                Rectangle(
                    (b_s, a_s), hb, ha,
                    fill=False,
                    edgecolor=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=1.0,
                    zorder=5
                )
            )
            # draw rb x ra rectangle (rows=rb, cols=ra) - symmetric block
            ax.add_patch(
                Rectangle(
                    (a_s, b_s), ha, hb,
                    fill=False,
                    edgecolor=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=1.0,
                    zorder=5
                )
            )
                
    # mark regions on all subplots
    for ax in axes.ravel():
        _mark_pairs(ax, pairs)
        # actually, N is the same for both structures
        N = dist_a.shape[0] if ax in (axes[0,0], axes[1,0]) else dist_b.shape[0]
        ax.set_xlim(0, N-1)
        ax.set_ylim(N-1, 0)
        # set labels , note that residue index is 0-based here
        ax.set_xlabel("Residue index (0-based)")
        ax.set_ylabel("Residue index (0-based)")

    # color bars are shared between subplots in the same row, cause we are comparing the same protein in structure1 and structure2
    cbar_top = fig.colorbar(imA, ax=[axes[0,0], axes[0,1]], fraction=0.046, pad=0.02)
    cbar_top.set_label("Cα–Cα distance (Å)")
    cbar_bot = fig.colorbar(imAB, ax=[axes[1,0], axes[1,1]], fraction=0.046, pad=0.02)
    cbar_bot.set_label("Δ distance (Å)")

    # save the figure in pdf format
    # or we can save it in png format if needed, set dpi=300 or higher
    if out_file:
        plt.savefig(out_file, bbox_inches='tight')

    
    # return the contact matrices if needed
    # first, we summary the current result 
    result = {
        "A":{
            "file": mmcif_file_a, # file path
            "N_res": Na, # number of residues of whole protein A
            "dist_matrix": dist_a, # array/matrix of pairwise distance of protein A
            "res_ca_info": ca_info_a # list of dicts of CA atom info for whole protein A
        },
        "B":{
            "file": mmcif_file_b,
            "N_res": Nb,
            "dist_matrix": dist_b,
            "res_ca_info": ca_info_b
        },
        "Dist_diff":{
            "A-B": diff_ab, # array/matrix of distance difference A - B
            "B-A": diff_ba # array/matrix of distance difference B - A
        },
        "pairs": pairs # list of region pairs compared
    }

    # then, for each region pair, we extract the sub-matrix and related info
    details = []
    for (ra, rb) in pairs:
        a_s, a_e = ra
        b_s, b_e = rb
        sub_a = dist_a[a_s:a_e+1, b_s:b_e+1]
        sub_b = dist_b[a_s:a_e+1, b_s:b_e+1]
        sub_ab = diff_ab[a_s:a_e+1, b_s:b_e+1]
        sub_ba = diff_ba[b_s:b_e+1, a_s:a_e+1]
        if sub_a.size == 0 or sub_b.size == 0:
            raise ValueError(f"The selected regions {ra} or {rb} are invalid, resulting in empty sub-matrix.")
        details.append({
            "pair": (ra, rb), # region pair
            "A_sub": sub_a, # sub-matrix of region ra vs region rb in structure A
            "B_sub": sub_b, # sub-matrix of region ra vs region rb in structure B
            "A-B_sub": sub_ab, # sub-matrix of region ra vs region rb in distance difference A - B
            "B-A_sub": sub_ba, # sub-matrix of region ra vs region rb in distance difference B - A
            "shape": sub_a.shape, # shape of the sub-matrix
            "mean_A_sub": float(np.nanmean(sub_a)), # mean value of the sub-matrix in structure A
            "mean_B_sub": float(np.nanmean(sub_b)), # mean value of the sub-matrix in structure B
            "mean_A-B_sub": float(np.nanmean(sub_ab)), # mean value of the sub-matrix in distance difference A - B
            "mean_B-A_sub": float(np.nanmean(sub_ba)) # mean value of the sub-matrix in distance difference B - A
        })
    result["Pairs_detail"] = details

    # finally, we return the result if needed
    if return_maxtrix:
        return result



