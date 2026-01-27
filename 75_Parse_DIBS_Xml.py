# https://dibs.pbrg.hu/downloads.php

import xml.etree.ElementTree as ET
import pandas as pd
import os

# Path to the XML file
file_path = '/data2/IDR-DomINTER/data/raw/DIBS/DIBS_complete.xml'

print(f"Parsing from scratch: {file_path}")
tree = ET.parse(file_path)
root = tree.getroot()

def get_text(parent, path):
    if parent is None: return None
    node = parent.find(path)
    return node.text if node is not None else None

def get_go_terms(parent, category):
    """Extracts GO terms for a specific category (molecular_function, etc.) as a string."""
    cat_node = parent.find(category)
    if cat_node is None:
        return None
    go_terms = []
    for go in cat_node.findall('go'):
        acc = get_text(go, 'accession')
        name = get_text(go, 'name')
        if acc and name:
            go_terms.append(f"{acc}:{name}")
    return "||".join(go_terms)

records = []

# Iterate through each Entry in the XML
for entry in root.findall('entry'):
    # --- 1. General Entry Metadata ---
    meta = {}
    meta['entry_accession'] = get_text(entry, 'accession')
    
    gen = entry.find('general')
    if gen is not None:
        meta['entry_name'] = get_text(gen, 'name')
        meta['pdb_id'] = get_text(gen, 'pdb_id')
        meta['exp_method'] = get_text(gen, 'exp_method')
        meta['source_organism'] = get_text(gen, 'source_organism')
        meta['disorder_status'] = get_text(gen, 'disorder_status')
        meta['kd_value'] = get_text(gen, 'kd/value') # Renamed slightly for clarity
        meta['kd_pubmed_id'] = get_text(gen, 'kd/pubmed_id')
        
        # Publication details
        pub = gen.find('publication')
        if pub is not None:
            meta['pub_pmid'] = get_text(pub, 'pmid')
            meta['pub_title'] = get_text(pub, 'title')
            meta['pub_year'] = get_text(pub, 'year')
            meta['pub_authors'] = get_text(pub, 'authors')

    # --- 1.5 Function (GO Terms) ---
    func = entry.find('function')
    if func is not None:
        meta['GO_Molecular_Function'] = get_go_terms(func, 'molecular_function')
        meta['GO_Biological_Process'] = get_go_terms(func, 'biological_process')
        meta['GO_Cellular_Component'] = get_go_terms(func, 'cellular_component')
    else:
        meta['GO_Molecular_Function'] = None
        meta['GO_Biological_Process'] = None
        meta['GO_Cellular_Component'] = None

    # --- 2. Evidence Mapping ---
    evidence_map = {}
    ev_node = entry.find('evidence')
    if ev_node is not None:
        for ce in ev_node.findall('chain_evidence'):
            c_id = get_text(ce, 'chain_id')
            sup = get_text(ce, 'support')
            if c_id:
                evidence_map[c_id] = sup

    # --- 3. Parse Chains (Separate IDR and Ordered) ---
    macro = entry.find('macromolecules')
    idr_chains = []
    ordered_chains = []
    domain_type = None
    nr_chains = None
    macro_note = None
    
    if macro is not None:
        macro_gen = macro.find('general')
        if macro_gen is not None:
            domain_type = get_text(macro_gen, 'domain_type')
            nr_chains = get_text(macro_gen, 'nr_of_chains')
            macro_note = get_text(macro_gen, 'note')
            
        for chain in macro.findall('chain'):
            c_id = get_text(chain, 'id')
            c_seq = get_text(chain, 'sequence')
            c_len = get_text(chain, 'length')
            
            # Extract UniProt Boundaries (The PRIMARY interaction site info)
            uniprot_node = chain.find('uniprot')
            uniprot_id = get_text(uniprot_node, 'id') if uniprot_node is not None else None
            uniprot_start = get_text(uniprot_node, 'start') if uniprot_node is not None else None
            uniprot_end = get_text(uniprot_node, 'end') if uniprot_node is not None else None
            uniprot_cov = get_text(uniprot_node, 'coverage') if uniprot_node is not None else None

            # Base Object
            chain_obj = {
                'id': c_id,
                'name': get_text(chain, 'name'),
                'type': get_text(chain, 'type'),
                'organism': get_text(chain, 'source_organism'),
                'sequence': c_seq,
                'length': c_len,
                'uniprot_id': uniprot_id,
                'uniprot_start': uniprot_start,
                'uniprot_end': uniprot_end,
                'uniprot_coverage': uniprot_cov,
                'evidence': evidence_map.get(c_id),
                'secondary_structures': [],
                'pfam_domains': [],
                'other_regions': []
            }
            
            # Extract Detailed Regions (Secondary Structures, Pfam, etc.)
            regs_node = chain.find('regions')
            if regs_node is not None:
                for r in regs_node.findall('region'):
                    r_type = get_text(r, 'region_type')
                    r_name = get_text(r, 'region_name')
                    r_start = get_text(r, 'region_start')
                    r_end = get_text(r, 'region_end')
                    
                    region_str = f"{r_name}({r_start}-{r_end})"
                    
                    if r_type == 'secondary structure':
                        chain_obj['secondary_structures'].append(region_str)
                    elif r_type == 'pfam':
                        chain_obj['pfam_domains'].append(region_str)
                    else:
                        chain_obj['other_regions'].append(f"{r_type}:{region_str}")
            
            if chain_obj['type'] == 'Disordered':
                idr_chains.append(chain_obj)
            else:
                ordered_chains.append(chain_obj)
    
    # --- 4. Flatten Ordered Partner(s) Info ---
    # Combine Partner infos (usually forming the binding surface)
    # We aggregate them using '||' delimiter if there are multiple ordered chains
    p_ids = "||".join([str(c['id']) for c in ordered_chains])
    p_names = "||".join([str(c['name']) for c in ordered_chains])
    p_uniprots = "||".join([str(c['uniprot_id']) for c in ordered_chains])
    p_seqs = "||".join([str(c['sequence']) for c in ordered_chains])
    p_lens = "||".join([str(c['length']) for c in ordered_chains])
    
    # Partner Boundaries (The Ordered Binding Site)
    p_starts = "||".join([str(c['uniprot_start']) for c in ordered_chains])
    p_ends = "||".join([str(c['uniprot_end']) for c in ordered_chains])
    
    # Partner Annotations
    p_sec_structs = "||".join([",".join(c['secondary_structures']) for c in ordered_chains])
    p_pfams = "||".join([",".join(c['pfam_domains']) for c in ordered_chains])
    
    # --- 5. Construct Rows based on IDR Chain ---
    # Create one row for each Disordered chain
    for idr in idr_chains:
        row = meta.copy()
        
        # --- MACRO METADATA ---
        row['macro_nr_chains'] = nr_chains
        row['macro_domain_type'] = domain_type
        # row['macro_note'] = macro_note # Optional, often verbose

        # --- IDR SIDE ---
        row['IDR_Chain_ID'] = idr['id']
        row['IDR_Name'] = idr['name']
        row['IDR_Organism'] = idr['organism']
        row['IDR_UniProt_ID'] = idr['uniprot_id']
        row['IDR_Length'] = idr['length']
        
        # THE CORE INTERACTION SITE (IDR)
        row['IDR_UniProt_Start'] = idr['uniprot_start']
        row['IDR_UniProt_End'] = idr['uniprot_end']
        row['IDR_Sequence'] = idr['sequence']
        row['IDR_UniProt_Coverage'] = idr['uniprot_coverage']
        
        row['IDR_Annotations'] = ",".join(idr['other_regions'] + idr['pfam_domains']) # Usually IDRs have motifs or linear regions
        row['IDR_Evidence'] = idr['evidence']

        # --- PARTNER (ORDERED) SIDE ---
        row['Partner_Chain_ID'] = p_ids
        row['Partner_Name'] = p_names
        row['Partner_Domain_Type'] = domain_type
        row['Partner_UniProt_ID'] = p_uniprots
        row['Partner_Length'] = p_lens
        
        # THE CORE INTERACTION SITE (PARTNER)
        row['Partner_UniProt_Start'] = p_starts
        row['Partner_UniProt_End'] = p_ends
        row['Partner_Sequence'] = p_seqs
        
        # DETAILED REGIONS
        row['Partner_Secondary_Structures'] = p_sec_structs
        row['Partner_Pfam_Domains'] = p_pfams

        records.append(row)

# Create final DataFrame
df_interactions = pd.DataFrame(records)
print(f"Successfully parsed {len(df_interactions)} interaction pairs.")

# Display relevant columns to verify structure
cols_to_show = [
    'entry_accession', 
    'IDR_UniProt_ID', 'IDR_UniProt_Start', 'IDR_UniProt_End',
    'Partner_UniProt_ID', 'Partner_UniProt_Start', 'Partner_UniProt_End',
    'Partner_Domain_Type', 'macro_nr_chains', 'GO_Molecular_Function'
]
df_interactions[cols_to_show].head()
