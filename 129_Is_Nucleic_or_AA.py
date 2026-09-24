# pdb/mmcif 判断某个残基是否是核酸还是氨基酸

# 1. 传统写法：自己写resname匹配
# 参考：https://github.com/osercinoglu/grinn/blob/a8b2fc1b464b113907436004f3c6cb4ad23c325f/grinn_workflow.py#L2992
PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "SEC",
    "PYL",
}
DNA_RESIDUES = {"DA", "DC", "DG", "DT", "DI"}
RNA_RESIDUES = {"A", "C", "G", "U", "I"}
NUCLEIC_RESIDUES = DNA_RESIDUES | RNA_RESIDUES

def is_water_molecule(mol_name):
    """Check if molecule name indicates water."""
    water_names = {'sol', 'water', 'wat', 'h2o', 'tip3', 'tip4', 'tip5', 'spc', 'spce'}
    return mol_name.lower() in water_names

def is_ion_molecule(mol_name):
    """Check if molecule name indicates an ion."""
    ion_names = {'na', 'cl', 'k', 'mg', 'ca', 'zn', 'fe', 'na+', 'cl-', 'k+', 'mg2+', 'ca2+', 'zn2+'}
    return mol_name.lower() in ion_names

def is_water_residue(res_name):
    """Check if residue is water, including potentially truncated names from GRO conversion."""
    water_names = {
        'SOL', 'WAT', 'H2O',  # Standard water names
        'TIP3', 'TIP4', 'TIP5', 'TIP',  # TIP water models (including truncated)
        'SPC', 'SPCE',  # SPC water models
        'OPC', 'OPC3'   # OPC water models
    }
    return res_name.upper() in water_names

def is_ion_residue(res_name):
    """Check if residue is an ion, including common ion names."""
    ion_names = {
        'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE',  # Common ions
        'NA+', 'CL-', 'K+', 'MG2+', 'CA2+', 'ZN2+',  # With charges
        'SOD', 'CLA', 'POT', 'MAG', 'CAL'  # Alternative names
    }
    return res_name.upper() in ion_names

def is_protein_residue(res_name):
    """Check if residue is a standard protein residue."""
    protein_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP'  # Histidine variants
    }
    return res_name in protein_residues

def is_nucleic_residue(res_name):
    """Check if residue is a nucleic acid residue."""
    nucleic_residues = {
        'A', 'T', 'G', 'C', 'U',  # Single letter
        'DA', 'DT', 'DG', 'DC',   # DNA
        'RA', 'RU', 'RG', 'RC',   # RNA
        'ADE', 'THY', 'GUA', 'CYT', 'URA'  # Full names
    }
    return res_name in nucleic_residues



####################################################################################################################

# 2. 调api写法
from Bio.PDB.Polypeptide import is_nucleic, is_aa
is_nucleic(residue), is_aa(residue)
