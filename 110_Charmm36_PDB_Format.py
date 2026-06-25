# convert DNA+protein structure in PDB format into another PDB format that charmm36 force field can accept

# 1. original version
# https://github.com/andrew065/Protein_DNA_MD/blob/main/src/utils/c36_pdb_format.py

from pathlib import Path
import sys

# Universal DNA rename map
common_map = {"OP1": "O1P", "OP2": "O2P"}

# Thymine-specific map
dt_map = {"C7": "C5M", "H71": "H5M1", "H72": "H5M2", "H73": "H5M3"}

def fix_pdb(infile, outfile):
    n_changes = 0
    out_lines = []

    for line in Path(infile).read_text().splitlines():
        if line.startswith(("ATOM","HETATM")):
            if len(line) < 80:
                line = line + " " * (80 - len(line))
            atom = line[12:16].strip()
            resn = line[17:20].strip()

            if resn in {"DA","DC","DG","DT"} and atom in common_map:
                new_atom = common_map[atom].rjust(4)
                line = line[:12] + new_atom + line[16:]
                n_changes += 1

            if resn == "DT" and atom in dt_map:
                new_atom = dt_map[atom].rjust(4)
                line = line[:12] + new_atom + line[16:]
                n_changes += 1

        # Don’t strip trailing spaces here; preserve exact column layout
        out_lines.append(line)

    Path(outfile).write_text("\n".join(out_lines) + "\n")
    print(f"Applied {n_changes} changes. Output written to {outfile}")

if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        print(__doc__)
        sys.exit(1)

    infile = Path(sys.argv[1])
    if len(sys.argv) == 3:
        outfile = Path(sys.argv[2])
    else:
        outfile = infile.with_name(f"{infile.stem}_c36.pdb")

    fix_pdb(infile, outfile)



###################################################################################################################

# 2. modified version

#!/usr/bin/env python3
"""
CHARMM36 PDB format converter for protein-DNA complexes.

Converts standard PDB atom naming to CHARMM36 conventions:
  - DNA phosphate: OP1 -> O1P, OP2 -> O2P, OP3 -> O3P
  - Thymidine methyl: C7 -> C5M, H71/H72/H73 -> H5M1/H5M2/H5M3
  - Sugar prime/star: * -> ' (e.g., O3* -> O3', H5*1 -> H5'1)
  - Sugar geminal hydrogens: H2'1 -> H2', H2'2 -> H2''; H5'1 -> H5', H5'2 -> H5''
  - Zn-coordinating CYS -> CYM (deprotonated cysteine for zinc finger)
  - Auto-assign HIS protonation states (HSD/HSE/HSP)
"""

import argparse
import re
import sys
from typing import Optional


def convert_atom_name(atom_name: str) -> str:
    """Convert a single atom name to CHARMM36 convention."""
    # Thymidine methyl group
    if atom_name == "C7":
        return "C5M"
    if atom_name in ("H71", "H72", "H73"):
        n = atom_name[-1]
        return f"H5M{n}"

    # DNA phosphate oxygens
    if atom_name == "OP1":
        return "O1P"
    if atom_name == "OP2":
        return "O2P"
    if atom_name == "OP3":
        return "O3P"

    # Sugar prime/star notation: O3* -> O3', H5*1 -> H5'1 etc.
    if "*" in atom_name:
        atom_name = atom_name.replace("*", "'")

    # Geminal hydrogen pairs on sugar
    # H2'1 -> H2', H2'2 -> H2'' ; H5'1 -> H5', H5'2 -> H5''
    if re.match(r"^H2'1$", atom_name):
        return "H2'"
    if re.match(r"^H2'2$", atom_name):
        return "H2''"
    if re.match(r"^H5'1$", atom_name):
        return "H5'"
    if re.match(r"^H5'2$", atom_name):
        return "H5''"

    return atom_name


def assign_his_protonation(
    pdb_lines: list[str], auto_his: bool = True, default: str = "HSD"
) -> list[str]:
    """
    Auto-assign HIS protonation state based on hydrogen positions.
    - HSD (delta tautomer): only HD1 present
    - HSE (epsilon tautomer): only HE2 present
    - HSP (doubly protonated): both HD1 and HE2 present
    - If neither present, use default
    """
    if not auto_his:
        return pdb_lines

    # Group atoms by residue
    his_residues: dict[str, dict[str, bool]] = {}

    for line in pdb_lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        res_name = line[17:20].strip()
        if res_name != "HIS":
            continue
        res_id = line[21:27]  # chain + resnum + insertion
        atom_name = line[12:16].strip()

        if res_id not in his_residues:
            his_residues[res_id] = {"HD1": False, "HE2": False}
        if atom_name == "HD1":
            his_residues[res_id]["HD1"] = True
        elif atom_name == "HE2":
            his_residues[res_id]["HE2"] = True

    # Determine protonation state for each HIS
    his_states: dict[str, str] = {}
    for res_id, hydrogens in his_residues.items():
        if hydrogens["HD1"] and hydrogens["HE2"]:
            his_states[res_id] = "HSP"
        elif hydrogens["HD1"]:
            his_states[res_id] = "HSD"
        elif hydrogens["HE2"]:
            his_states[res_id] = "HSE"
        else:
            his_states[res_id] = default

    # Apply the changes
    result = []
    for line in pdb_lines:
        if not line.startswith(("ATOM", "HETATM")):
            result.append(line)
            continue
        res_name = line[17:20].strip()
        if res_name != "HIS":
            result.append(line)
            continue
        res_id = line[21:27]
        new_name = his_states.get(res_id, default)
        result.append(line[:17] + f"{new_name:<3}" + line[20:])

    return result


def convert_cys_to_cym(pdb_lines: list[str], zn_cutoff: float = 3.0) -> list[str]:
    """
    Convert Zn-coordinating CYS residues to CYM (deprotonated cysteine).
    A CYS is considered Zn-coordinating if its SG atom is within zn_cutoff
    of any ZN atom.
    """
    # Find all ZN atoms
    zn_positions: list[list[float]] = []
    for line in pdb_lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        res_name = line[17:20].strip()
        if res_name == "ZN" or res_name == "ZN2":
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            zn_positions.append([x, y, z])

    if not zn_positions:
        return pdb_lines

    # Find CYS SG atoms near ZN
    coordinating_cys: set[str] = set()  # (chain, resnum, insertion)
    for line in pdb_lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        res_name = line[17:20].strip()
        if res_name != "CYS":
            continue
        atom_name = line[12:16].strip()
        if atom_name != "SG":
            continue
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])

        for zn_pos in zn_positions:
            dist = ((x - zn_pos[0]) ** 2 + (y - zn_pos[1]) ** 2 + (z - zn_pos[2]) ** 2) ** 0.5
            if dist < zn_cutoff:
                res_id = line[21:27]
                coordinating_cys.add(res_id)
                break

    # Rename coordinating CYS to CYM, remove HG
    result = []
    for line in pdb_lines:
        if not line.startswith(("ATOM", "HETATM")):
            result.append(line)
            continue
        res_name = line[17:20].strip()
        if res_name == "CYS":
            res_id = line[21:27]
            if res_id in coordinating_cys:
                atom_name = line[12:16].strip()
                if atom_name == "HG":
                    continue  # remove HG for deprotonated CYM
                result.append(line[:17] + "CYM" + line[20:])
            else:
                result.append(line)
        else:
            result.append(line)

    return result


def format_pdb_columns(line: str) -> str:
    """
    Ensure strict PDB column widths.
    Atom name right-justified to column 16 (indices 12-15).
    """
    if not line.startswith(("ATOM", "HETATM")):
        return line
    atom_name = line[12:16].strip()
    # Right-justify within 4 chars at positions 13-16
    return line[:12] + f"{atom_name:>4}" + line[16:]


def process_pdb(input_path: str, output_path: str, auto_his: bool = True) -> None:
    with open(input_path) as f:
        pdb_lines = f.readlines()

    result = []
    for line in pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            new_name = convert_atom_name(atom_name)
            if new_name != atom_name:
                line = line[:12] + f"{new_name:>4}" + line[16:]
        result.append(line)

    # Auto-assign HIS protonation
    result = assign_his_protonation(result, auto_his=auto_his)

    # Convert Zn-coordinating CYS to CYM
    result = convert_cys_to_cym(result)

    # Ensure proper column formatting
    result = [format_pdb_columns(line) for line in result]

    with open(output_path, "w") as f:
        f.writelines(result)

    print(f"Converted {input_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDB to CHARMM36 naming conventions for protein-DNA complexes"
    )
    parser.add_argument("input", help="Input PDB file")
    parser.add_argument("-o", "--output", required=True, help="Output PDB file")
    parser.add_argument(
        "--auto-his",
        action="store_true",
        default=True,
        help="Auto-assign HIS protonation states (HSD/HSE/HSP)",
    )
    parser.add_argument(
        "--no-auto-his",
        dest="auto_his",
        action="store_false",
        help="Disable auto HIS protonation assignment",
    )
    args = parser.parse_args()

    process_pdb(args.input, args.output, auto_his=args.auto_his)


if __name__ == "__main__":
    main()
