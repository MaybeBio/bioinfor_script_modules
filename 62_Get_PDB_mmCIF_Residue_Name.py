# 获取PDB或者是mmCIF格式文件中残基的具体分子类型

# 1, 
protein = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K','ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N','GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W','ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', "UNK": "X"}
dna = {'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T'}
rna = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U'}

# 然后我们可以直接抽取第1个残基的name来确定该chain或者是原子的具体类型
def get_type(self):
	first_res = self.child_list[0].resname.strip()  # Get the first residue name to see what kind of sequence it is
	if first_res in self.protein: #if the first residue is in the protein dictionary, we have a protein sequence
		return "protein"
		
	elif first_res in self.dna: #if the first residue is in the dna dictionary, we have a dna sequence
		return "dna"

	else:
		return "rna"

============================================================================================================================================

# 2,
# 1. Protein -> CA
if Polypeptide.is_aa(res, standard=True):
	if 'CA' in res:
		rep_atom = res['CA']
        rtype = "Protein"
                
# 2. Nucleic Acid -> C1' (Distinguish DNA/RNA)
elif "C1'" in res:
	rep_atom = res["C1'"]
	rname = res.resname.strip()
                    
	if rname in {'DA', 'DC', 'DG', 'DT'}:
		rtype = "DNA"
    elif rname in {'A', 'C', 'G', 'U'}:
        rtype = "RNA"
    else:
        rtype = "Nucleic" # Fallback for modified bases
                
# 3. Others (Ligands/Ions) -> First Atom
else:
	atoms = list(res.get_atoms())
    if len(atoms) > 0:
		rep_atom = atoms[0]
        rtype = "Ligand
