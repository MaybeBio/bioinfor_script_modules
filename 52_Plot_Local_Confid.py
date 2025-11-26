# 绘制AlphaFold3输出的PAE矩阵热图

def plot_local_confidence(full_json_file_path, output_path=None, chains: Optional[str]=None , tick_step: int = 200):
    """
    Plot PAE matrix with chain/top/left tracks.

    Parameters:
      full_json_file_path: path to full_data_X.json (contains 'pae', 'token_chain_ids', 'token_res_ids', ...)
      output_path: if provided, save figure into this directory (filename: local_confidence_pae.png)
      chain: Optional chain id (e.g. "A") to restrict view to that chain only; default None = show all chains
      tick_step: residue tick spacing (used when token_res_ids are raw residue numbers); ticks will include residue==1 and every tick_step
    """
    local_confidence = load_json_data(full_json_file_path)
    if local_confidence is None:
        raise ValueError("Failed to load JSON data.")

    # extract local confidence measures
    pae_matrix = np.asarray(local_confidence['pae'], dtype=float) # 2D ndarray
    contact_probs = np.asarray(local_confidence.get('contact_probs', []), dtype=float) if 'contact_probs' in local_confidence else None
    atom_chain_ids = np.asarray(local_confidence.get('atom_chain_ids', []))  # list of chain IDs for each residue
    atom_plddts = np.asarray(local_confidence.get('atom_plddts', []), dtype=float)  # list of pLDDT scores for each residue
    token_chain_ids = np.asarray(local_confidence['token_chain_ids'])  # list of chain IDs for each residue
    token_res_ids = np.asarray(local_confidence['token_res_ids'])  # list of residue indices

    # figure out the chain ranges for each chain in token_chain_ids
    def process_token_chains(token_chain_ids):
        unique_chain_ids = np.unique(token_chain_ids)
        chain_to_index = { chain_id: i for i, chain_id in enumerate(unique_chain_ids)}
        token_chain_indices = np.asarray([chain_to_index[chain_id]  for chain_id in token_chain_ids]).reshape(1,-1)
        chain_to_range: Dict[str, Tuple[int, int]] = {}
        chain_index_to_range: Dict[int, Tuple[int, int]] = {}
        for chain_id in unique_chain_ids:
            indices = np.where(token_chain_ids == chain_id)[0]
            chain_to_range[chain_id] = (indices[0], indices[-1])
            chain_index_to_range[chain_to_index[chain_id]] = (indices[0], indices[-1])
        return chain_to_index, chain_to_range, chain_index_to_range, token_chain_indices

    chain_to_index, chain_to_range, chain_index_to_range, token_chain_indices = process_token_chains(token_chain_ids)

    # if user requested specific chain(s), slice matrices/vectors to those chain regions
    # chain may be: None (all), a single string "A", or a sequence/list ["A","B"]
    sel_chains = [chains] if isinstance(chains, str) else (list(chains) if chains is not None else None)
    if sel_chains is not None:
        missing = [c for c in sel_chains if c not in chain_to_range]
        # validate
        if missing:
            raise ValueError(f"Chain(s) not found: {missing}")
        # get indices (preserve original token order) for any token whose chain in selected_chains
        mask = np.isin(token_chain_ids, sel_chains)
        idx = np.where(mask)[0]
        if idx.size == 0:
            raise ValueError("No residues for selected chain(s).")
        # slice PAE and auxiliary arrays to selected indices
        pae_matrix = pae_matrix[np.ix_(idx, idx)]
        token_chain_indices = token_chain_indices[:, idx]
        token_res_ids = token_res_ids[idx]
        # rebuild chain_to_range for the subset: detect contiguous blocks per chain in the subset
        chain_to_range = {}
        start = None
        cur = None
        prev = None
        for rel, abs_i in enumerate(idx):
            ch = token_chain_ids[abs_i]
            if cur is None:
                cur = ch; start = prev = rel
            elif ch != cur:
                chain_to_range[cur] = (start, prev)
                cur = ch; start = prev = rel
            else:
                prev = rel
        if cur is not None:
            chain_to_range[cur] = (start, prev)
    else:
        chain_to_range_start = {cid: s for cid, (s, e) in chain_to_range.items()}
        chain_to_end_index = {cid: e for cid, (s, e) in chain_to_range.items()}


    # Prepare figure
    fig, ax = plt.subplots(figsize=(10, 10))

    def hide_axes_frame(ax_in):
        """
        Hide the frame of the given matplotlib axis (used for top/left small axes).
        """
        for spine in ax_in.spines.values():
            spine.set_visible(False)
        ax_in.set_xticks([])
        ax_in.set_yticks([])

    # compute tick positions and labels based on token_res_ids (after possible slicing)
    xticks_loc = []
    xticks_present = []
    for i in range(len(token_res_ids)):
        if token_res_ids[i] == 1 or (tick_step and token_res_ids[i] % tick_step == 0):
            xticks_loc.append(i)
            xticks_present.append(int(token_res_ids[i]))

    # display PAE matrix
    image = ax.imshow(pae_matrix, cmap="Greens_r", origin='upper', aspect='auto')
    # Set numeric ticks for both axes
    ax.set_xticks(xticks_loc)
    ax.set_xticklabels(xticks_present, fontsize=10)
    ax.set_yticks(xticks_loc)
    ax.set_yticklabels(xticks_present, fontsize=10)
    ax.set_xlabel('Scored Residue', fontsize=10)
    ax.set_ylabel('Aligned Residue', fontsize=10)

    # set the frame to dashed line
    for spine in ax.spines.values():
        spine.set_linestyle("--")
        spine.set_linewidth(1)
        spine.set_color("k")

    # Create side axes
    divider = make_axes_locatable(ax)
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.2)
    ax_topbar = divider.append_axes("top", size="8%", pad=0.03)
    ax_leftbar = divider.append_axes("left", size="8%", pad=0.03)

    # topbar and leftbar use token_chain_indices (may have been sliced)
    ax_topbar.imshow(token_chain_indices, cmap="tab10", aspect="auto", alpha=0.7)
    hide_axes_frame(ax_topbar)

    ax_leftbar.imshow(token_chain_indices.T, cmap="tab10", aspect="auto", alpha=0.7)
    hide_axes_frame(ax_leftbar)

    # colorbar
    colorbar = fig.colorbar(image, cax=ax_colorbar, label="Expected Position Error (Å)")

    # If showing full chains, plot separators and labels; if sliced to one chain, annotate that chain only
    if chain is None:
        for chain_id,start in {cid: rng[0] for cid, rng in chain_to_range.items()}.items():
            if start != 0:
                ax.axhline(start - 0.5, color="k", linewidth=1, linestyle="--")
                ax.axvline(start - 0.5, color="k", linewidth=1, linestyle="--")
                ax_topbar.axvline(start - 0.5, color="w", linewidth=1, linestyle="-")
                ax_leftbar.axhline(start - 0.5, color="w", linewidth=1, linestyle="-")

        # Adding text annotations at the center of each token chain
        for chain_id, (start, end) in chain_to_range.items():
            start_index = start
            end_index = end
            center_index = (start_index + end_index) / 2
            ax_topbar.text(center_index, 0, chain_id, color='#222222', ha='center', va='center')
            ax_leftbar.text(0, center_index, chain_id, color='#222222', ha='center', va='center')
    else:
        # single-chain label in tracks
        ax_topbar.text((token_chain_indices.shape[1]-1)/2, 0, chain, color='#222222', ha='center', va='center')
        ax_leftbar.text(0, (token_chain_indices.shape[1]-1)/2, chain, color='#222222', ha='center', va='center')

    # save or show
    if output_path:
        plt.savefig(f"{output_path}/local_confidence_pae.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    return fig
