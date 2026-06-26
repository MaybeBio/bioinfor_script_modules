def prepare_pdb(config: ZF_DNAConfig) -> Path:
    """
    Clean protein-DNA PDB file: skip crystallographic waters, remove alternate conformations, remove 5' phosphate if using Amber FF force field, etc.
    结构预处理, 去除结晶水, 移除替代构型/多构象, 移除5'磷酸原子(如果使用Amber 力场)等.

    Notes
    - 1. 用于核酸的标准 Amber 力场并未提供带有 5'磷酸的残基模板, 通常需要手动移除 5'-末端的 P 和 O1P/O2P 原子. 
    从末端残基中移除任何磷酸基团；这些基团通常是通过合成产生的，但不是力场参数化时的典型方式
    https://gromacs.bioexcel.eu/t/atom-p-in-residue-da-90-was-not-found-in-rtp-entry-da5-with-30-atoms-while-sorting-atoms/7617/6
    https://gromacs.bioexcel.eu/t/atom-op3-in-residue-da-1-was-not-found-in-rtp-entry-da5-with-30-atoms-while-sorting-atoms/9570 
    https://gromacs.bioexcel.eu/t/amber-forcefield/5675
    http://gromacs.bioexcel.eu/t/atom-p-in-residue-dt-9-was-not-found-in-rtp-entry-dt5/7076/4
    参考: https://github.com/lacoperon/PDBRemovePhosphates
    - 2. 
    """
    
    raw_pdb = config.zf_base / config.pdb_file                                        
    clean_pdb = config.project_dir / f"{Path(config.pdb_file).stem}_clean.pdb"        
                                                                                        
    logging.info("="*60)                                                              
    logging.info("STEP 1: Preparing PDB file")                                        
    logging.info("="*60)                                                              
                                                                                        
    assert_file(raw_pdb, "Raw PDB file")                                                                                                                                                                                         

    with open(raw_pdb) as f:
        lines_out = []
        after_ter = True  # Start with True to handle first residue as potential chain start
        after_resid = None
        phosphate_removed = 0

        for line in f:                                                                
            # Skip crystallographic waters                                                
            if line.startswith("HETATM") and "HOH" in line:                               
                continue                                                                  
                                                                                            
            # Handle ATOM/HETATM lines                                                    
            if line.startswith("ATOM") or line.startswith("HETATM"):                      
                altloc = line[16]                                                         
                                                                                            
                # Remove alternate conformations (keep A only)                            
                if altloc not in (' ', 'A', ''):                                          
                    continue                                                              
                if altloc == 'A':                                                         
                    line = line[:16] + ' ' + line[17:]                                    
                                                                                            
                # Track residue ID after TER                                              
                resid = line[21:26].strip() # A   1                                                      
                resname = line[17:20].strip() # PRO DA                                            
                atom_name = line[13:16].strip() # N1 OP1                                         

                # Track the first residue after TER to identify chain starts                                                                            
                if after_ter:    
                    # This is the first residue after a TER record, which indicates the start of a new chain. We will use this information to determine if we are at the start of a DNA chain and potentially remove phosphate atoms.                                                         
                    after_resid = resid                                                   
                    after_ter = False                                                     
                                                                                            
                # Remove phosphate atoms from DNA chains at chain start                   
                is_dna = resname in ['DA', 'DT', 'DG', 'DC']                              
                is_phosphate = atom_name in ['P', 'OP1', 'OP2', 'OP3']                    

                # If we are at the first residue of a DNA chain (after TER), and the current atom is a phosphate atom, we will remove it. This is because standard Amber force fields do not have parameters for 5' phosphates, and they are typically not included in simulations.                                                                            
                # DNA链第1个残基的磷酸基团原子(5'末端磷酸基团)
                if after_resid == resid and is_dna and is_phosphate:                      
                    phosphate_removed += 1                                                
                    continue  # Skip this line (delete phosphate atom)                    
                                                                                            
                lines_out.append(line)                                                    
            else:                                                                         
                # Handle TER records                                                      
                if "TER" in line:                                                         
                    after_ter = True                                                      
                    lines_out.append(line)                                                
                else:                                                                     
                    # Keep all other lines (HEADER, CRYST1, END, etc.)                    
                    lines_out.append(line)                                                
                                                                                        
    with open(clean_pdb, 'w') as f:                                                   
        f.writelines(lines_out)                                                       
                                                                                        
    logging.info(f"  Clean PDB written: {clean_pdb}")                                 
    logging.info(f"  Total lines: {len(lines_out)}")                                  
    logging.info(f"  Phosphate atoms removed: {phosphate_removed}")                   
                                                                                        
    # Verify zinc ions present                                                        
    zn_count = sum(1 for l in lines_out if 'ZN' in l and l.startswith('HETATM'))      
    logging.info(f"  Zinc ions: {zn_count}")                                          
    if zn_count != 9:                                                                 
        logging.warning(f"  Expected 9 Zn ions for ZNF263 ZF1-9, found {zn_count}")   
                                                                                        
    return clean_pdb
