# 主要是收集给定1个DNA结合蛋白, 其对应的motif数据, 以及潜在的DNA结合位点序列数据
# 其实这两个数据是互补的: 有了motif数据(一般是构建的PWM矩阵, 就可以scan全基因组序列然后获取相应的DNA结合位点;
# 另外一方面, 有了DNA结合位点序列数据, 就可以构建PWM矩阵也就是motif数据

# 1, https://github.com/davidduclam/jaspar-tf-scanner

# (1) src/modules/binding_site_finder.py
# ---- SCAN DNA SEQUENCE FOR BINDING SITES ----
def scan_sequence(dna_sequence, pwm):
   """Scans DNA sequence for potential binding sites using PWM scoring."""
   seq_length = len(dna_sequence)
   motif_length = pwm.shape[1]
   scores = []

   for i in range(seq_length - motif_length + 1):
       subseq = dna_sequence[i : i + motif_length]
       score = sum(pwm["ACGT".index(base), j] for j, base in enumerate(subseq))
       scores.append((i, score))

   return scores

# (2) src/modules/jaspar_api.py

import streamlit as st
import requests
from constants import JASPAR_BASE_URL

# ---- JASPAR API FUNCTIONS ----
def search_jaspar(tf_name):
    """Search for transcription factor motifs by name."""
    try:
        response = requests.get(f"{JASPAR_BASE_URL}/?name={tf_name}&format=json")
        response.raise_for_status()
        motifs = response.json()
        return motifs["results"]
    except requests.RequestException as e:
        st.error(f"Error fetching data from JASPAR: {e}")
        return None

def fetch_jaspar_motif(jaspar_id):
    """Fetches motif matrix from JASPAR."""
    try:
        response = requests.get(f"{JASPAR_BASE_URL}/{jaspar_id}/?format=json")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching motif data from JASPAR: {e}")
        return None

# (3) src/modules/pfm_pwm_converter.py

import numpy as np

# ---- PFM TO PWM CONVERSION ----
def pfm_to_pwm(pfm_dict):
    """Convert a Position Frequency Matrix (PFM) to a PWM using Laplace smoothing"""
    pfm_array = np.array([pfm_dict[base] for base in "ACGT"], dtype=float)

    bg = np.array([0.25, 0.25, 0.25, 0.25])  # Background frequency for A, C, G, T
    total = np.sum(pfm_array, axis=0)  # Sum across all bases at each position

    # Adding pseudocount to avoid zero probabilities
    probabilities = (pfm_array + bg[:, None]) / (total + 1)

    pwm = np.log2(probabilities / bg[:, None])  # Log-odds conversion
    return pwm
