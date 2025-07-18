"""
Script to make a dataset of proteins containing their IDR sequences, 
features extracted from their sequences, and the log partition ratio, 
which is the value to be predicted
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ast
from localcider.sequenceParameters import SequenceParameters
import argparse


def get_SCD(sequence):
    """Function to calculate the SCD of a given protein sequence
    
    Inputs:
    sequence (string): Amino acid sequence of the protein
    Outputs:
    scd (float): SCD of the sequence
    """
    sequence_length = len(sequence)
    # SCD原始文献中只考虑KR带正电，此处补充H, 暂时不补充
    pos_charges = [1.0 if amino_acid == 'R' or amino_acid == 'K' else 0.0 for amino_acid in sequence]
    neg_charges = [-1.0 if amino_acid == 'D' or amino_acid == 'E' else 0.0 for amino_acid in sequence]
    charges = [pos_charges[i] + neg_charges[i] for i in range(sequence_length)]
    scd = 1.0/sequence_length*np.sum([np.sum([charges[i]*charges[j]*np.sqrt(j-i) 
                                              for j in range(i+1, sequence_length)]) for i in range(sequence_length)])
    return scd

def featurize_data(df, partition_ratios_df):
    """Function that extracts features from the IDRs present in each protein
    
    Inputs:
    -------
    df (pandas dataframe): Containing the list of IDR sequences in a protein
    partition_ratios_df (pandas dataframe): Containing the log partition ratios of the IDR sequences
    
    Outputs:
    --------
    df (pandas dataframe): Modified version of the same dataframe containing IDR features
    """
    
    # Create the dataset containing protein sequences, 
    # features extracted from their sequence, and 
    # log partition ratios (the value to be predicted) 

    num_points = df['IDR Sequence List'].size
    print("Size of the dataset: ", str(num_points), " IDRs ....")

    # IDR sequences in the data frame
    df['IDR Sequence Combined'] = np.array(['']*df['IDR Sequence List'].size)

    # Features from IDRs
    df['IDR Count'] = np.zeros(num_points)
    df['Total IDR Length'] = np.zeros(num_points)
    # Features from localcider
    df['Fraction Positive'] = np.zeros(num_points)
    df['Fraction Negative'] = np.zeros(num_points) 
    df['Fraction Expanding'] = np.zeros(num_points)
    df['FCR'] = np.zeros(num_points)
    df['NCPR'] = np.zeros(num_points)
    df['Kappa'] = np.zeros(num_points)
    df['Omega'] = np.zeros(num_points)
    df['Isoelectric Point'] = np.zeros(num_points)
    df['Uversky Hydropathy'] = np.zeros(num_points)
    df['PPII Propensity'] = np.zeros(num_points)
    df['Delta'] = np.zeros(num_points)
    df['Delta Max'] = np.zeros(num_points)
    # Feature: Sequence charge decoration
    df['SCD'] = np.zeros(num_points)

    # Value to be regressed
    df['Log Partition Ratios'] = np.zeros(df['IDR Sequence List'].size)

    # Construct the dataset
    for idx in range(df['Uniprot ID'].size):
        # Combine the IDR sequences and get their length
        for idr_sequence in df['IDR Sequence List'].iloc[idx]:
            df['IDR Sequence Combined'].iloc[idx] += str(idr_sequence)
            df['IDR Count'].iloc[idx] += 1
        df['Total IDR Length'].iloc[idx] = len(df['IDR Sequence Combined'].iloc[idx]) 
        # Get sequence features associated with charge patterning from localcider
        obj = SequenceParameters(df['IDR Sequence Combined'].iloc[idx])
        df['Fraction Positive'].iloc[idx] = obj.get_fraction_positive()
        df['Fraction Negative'].iloc[idx] = obj.get_fraction_negative()
        df['Fraction Expanding'].iloc[idx] = obj.get_fraction_expanding(pH=7.2)
        df['FCR'].iloc[idx] = obj.get_FCR(pH=7.2)
        df['NCPR'].iloc[idx] = obj.get_NCPR(pH=7.2)
        df['Omega'].iloc[idx] = obj.get_Omega()
        df['Kappa'].iloc[idx] = obj.get_kappa()
        df['Isoelectric Point'].iloc[idx] = obj.get_isoelectric_point()
        df['Uversky Hydropathy'].iloc[idx] = obj.get_uversky_hydropathy()
        df['PPII Propensity'].iloc[idx] = obj.get_PPII_propensity(mode='hilser')
        df['Delta'].iloc[idx] = obj.get_delta()
        df['Delta Max'].iloc[idx] = obj.get_deltaMax()
        # Get sequence charge decoration
        df['SCD'].iloc[idx] = get_SCD(df['IDR Sequence Combined'].iloc[idx])
        print('Finished featurizing ... ', str(idx+1) ,' IDRs ...')
        
        # Add the log partition ratio values
        df['Log Partition Ratios'].iloc[idx] = partition_ratios_df[partition_ratios_df['Uniprot ID'] 
                                                                    == df['Uniprot ID'].iloc[idx]]['Log2(average_P/S)']
    
    return df

if __name__ == "__main__":
    
    # Read command line arguments 
    parser = argparse.ArgumentParser(description='Run script to generate dataset of IDR features')
    parser.add_argument('--i', help="Name of input file containing lists of IDRs in each protein", required=True)
    parser.add_argument('--p', help="Name of input file contaning log partition ratios", required=True)
    parser.add_argument('--o', help="Name of output file to export the data matrix into", required=True)
    args = parser.parse_args()
    
    idr_file_name = args.i
    pr_file_name = args.p
    output_file_name = args.o
    
    # Import dataset containing IDR sequences
    df = pd.read_csv(idr_file_name)
    # Convert the list of sequences from a string to a list 
    df['IDR Sequence List'] = [ast.literal_eval(entry) for entry in df['IDR Sequence List']]
    # Exclude proteins that have no IDRs:
    # Define a function to check if the element is an empty list
    def is_empty_list(lst):
        return len(lst) == 0
    # Drop rows where IDR Sequence List is an empty list
    df = df[~df['IDR Sequence List'].apply(is_empty_list)]
    
    # Import dataset containing log partition ratios
    partition_ratios_df = pd.read_csv(pr_file_name)
    
    # Featurize data
    df = featurize_data(df, partition_ratios_df)
    
    # Save data matrix
    df.head(10)
    df.to_csv(output_file_name, index=False)  
    
    

