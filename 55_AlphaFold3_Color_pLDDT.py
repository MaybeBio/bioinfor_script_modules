# 为alphafold3输出结果, 在pymol中依据plddt分数对结构上色可视化
# 参考修改自: https://github.com/cbalbin-bio/pymol-color-alphafold
# 

# 1,
from pymol import cmd

# Updated pLDDT bin colors (very low → very high)
# Very high (>90), High (70–90], Low (50–70], Very low (<=50)
_PLDDT_RGBs = [
    (0.992, 0.490, 0.302),  # very low (index 0)
    (0.996, 0.851, 0.212),  # low
    (0.416, 0.796, 0.945),  # high
    (0.051, 0.341, 0.827),  # very high
]

# Bin boundaries (lower inclusive; upper exclusive except last handled separately)
# B-factor (pLDDT) ranges:
#   [0,50], (50,70], (70,90], (90,100]
_BIN_LOWER = [0, 50, 70, 90]
_BIN_UPPER = [50, 70, 90, 100]  # 100 marks upper bound of top bin

def af3_plddt_bins(selection="all"):
    """
    Color atoms by pLDDT stored in B-factor using four bins:
      <=50 very low, 50-70 low, 70-90 high, >90 very high.
    Usage: af3_plddt_bins <object_or_selection>
    """
    base = f"({selection})"
    for i, (lo, up) in enumerate(zip(_BIN_LOWER, _BIN_UPPER)):
        rgb = _PLDDT_RGBs[i]
        # Define color name
        cname = f"plddt_bin_{i+1}"
        cmd.set_color(cname, rgb)

        # Selection logic avoiding <= / >= (use > and not > upper)
        if up is None:
            # Top bin: >90
            sel_expr = f"{base} and b>{lo}"
        elif i == 0:
            # First bin: <=50 -> not b>50
            sel_expr = f"{base} and (not b>{up})"
        else:
            # Middle bins: (lo, up] -> b>lo and not b>up
            sel_expr = f"{base} and b>{lo} and (not b>{up})"

        group_name = f"{selection}_plddt_{lo}_{up if up is not None else 'max'}"
        cmd.select(group_name, sel_expr)
        cmd.color(cname, group_name)

cmd.extend("af3_plddt_bins", af3_plddt_bins)
cmd.auto_arg[0]["af3_plddt_bins"] = [cmd.object_sc, "object", ""]
# Optionally auto-run:
# af3_plddt_bins()


==========================================================================================================================================================
# 2, 
# For open source Pymol installations, please refer to https://github.com/schrodinger/pymol-open-source

from pymol import cmd

# Define custom pLDDT colors (RGB in 0–1 range)
cmd.set_color("plddt_vh", [0.051, 0.341, 0.827])  # Very high (>90)
cmd.set_color("plddt_h",  [0.416, 0.796, 0.945])  # High (90 >= x > 70)
cmd.set_color("plddt_l",  [0.996, 0.851, 0.212])  # Low  (70 >= x > 50)
cmd.set_color("plddt_vl", [0.992, 0.490, 0.302])  # Very low (<=50)

def af3_color_plddt(selection="all"):
    """
    Color AlphaFold(3) structures by pLDDT (stored in B-factor).
    Usage: af3_color_plddt sele
    """

    # Use 'between a, b' (inclusive) to avoid '<=' parser issues
    cmd.color("plddt_vh", f"({selection}) and b>90")
    cmd.color("plddt_h",  f"({selection}) and (b>70 and not b>90)")  # ≈ 70<b<=90
    cmd.color("plddt_l",  f"({selection}) and (b>50 and not b>70)")  # ≈ 50<b<=70
    cmd.color("plddt_vl", f"({selection}) and not b>50")             # ≈ b<=50

    '''
    cmd.color("plddt_vh", f"({selection}) and b > 90")
    cmd.color("plddt_h",  f"({selection}) and b <= 90 and b > 70")
    cmd.color("plddt_l",  f"({selection}) and b <= 70 and b > 50")
    cmd.color("plddt_vl", f"({selection}) and b <= 50")
    '''

# register PyMOL command
cmd.extend("af3_color_plddt", af3_color_plddt)
cmd.auto_arg[0]["af3_color_plddt"] = [cmd.object_sc, "object", ""]
