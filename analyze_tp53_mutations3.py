import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# Reference sequence for ENST00000269305.4
mRNA = ("ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGAAGCTCCCAGAATGCCAGAGGCTGCTCCCCGCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCCCCCTCCTGGCCCCTGTCATCTTCTGTCCCTTCCCAGAAAACCTACCAGGGCAGCTACGGTTTCCGTCTGGGCTTCTTGCATTCTGGGACAGCCAAGTCTGTGACTTGCACGTACTCCCCTGCCCTCAACAAGATGTTTTGCCAACTGGCCAAGACCTGCCCTGTGCAGCTGTGGGTTGATTCCACACCCCCGCCCGGCACCCGCGTCCGCGCCATGGCCATCTACAAGCAGTCACAGCACATGACGGAGGTTGTGAGGCGCTGCCCCCACCATGAGCGCTGCTCAGATAGCGATGGTCTGGCCCCTCCTCAGCATCTTATCCGAGTGGAAGGAAATTTGCGTGTGGAGTATTTGGATGACAGAAACACTTTTCGACATAGTGTGGTGGTGCCCTATGAGCCGCCTGAGGTTGGCTCTGACTGTACCACCATCCACTACAACTACATGTGTAACAGTTCCTGCATGGGCGGCATGAACCGGAGGCCCATCCTCACCATCATCACACTGGAAGACTCCAGTGGTAATCTACTGGGACGGAACAGCTTTGAGGTGCGTGTTTGTGCCTGTCCTGGGAGAGACCGGCGCACAGAGGAAGAGAATCTCCGCAAGAAAGGGGAGCCTCACCACGAGCTGCCCCCAGGGAGCACTAAGCGAGCACTGCCCAACAACACCAGCTCCTCTCCCCAGCCAAAGAAGAAACCACTGGATGGAGAATATTTCACCCTTCAGATCCGTGGGCGTGAGCGCTTCGAGATGTTCCGAGAGCTGAATGAGGCCTTGGAACTCAAGGATGCCCAGGCTGGGAAGGAGCCAGGGGGGAGCAGGGCTCACTCCAGCCACCTGAAGTCCAAAAAGGGTCAGTCTACCTCCCGCCATAAAAAACTCATGTTCAAGACAGAAGGGCCTGACTCAGACTGAcattctccacttcttgttccccactgacagcct")

# Load TSV file
try:
    df = pd.read_csv("table (1).tsv", sep="\t", low_memory=False)
except FileNotFoundError:
    print("Error: 'table (1).tsv' not found. Please check the file path.")
    exit(1)
except Exception as e:
    print(f"Error loading TSV file: {e}")
    exit(1)

# Exclude specified mutation types
excluded_types = ["Splice_Site", "Splice_Region", "Fusion"]
filtered_df = df[~df["Mutation Type"].isin(excluded_types)]

# Split into SNP and DEL/INS
snp_df = filtered_df[filtered_df["Variant Type"] == "SNP"]
delins_df = filtered_df[filtered_df["Variant Type"].isin(["DEL", "INS", "DELINS", "DUP"])]

# Function to check for AA motif (consecutive A's)
def has_aa(seq):
    seq = seq.upper()
    return "AA" in seq

# Function to check for AXA motif (e.g., AGA, ACA, ATA)
def has_axa(seq):
    seq = seq.upper()
    for i in range(len(seq) - 2):
        if seq[i] == 'A' and seq[i + 2] == 'A':
            return True
    return False

# Function to check for exactly one A (no AA or AXA)
def has_single_a(seq):
    seq = seq.upper()
    a_count = seq.count('A')
    return a_count == 1 and not has_aa(seq) and not has_axa(seq)

# Function to check for any A (Single A, AA, or AXA)
def has_any_a(seq):
    seq = seq.upper()
    return seq.count('A') > 0

# Function to check for AA or AXA (for existing Category B)
def has_aa_or_axa(seq):
    return has_aa(seq) or has_axa(seq)

# --- Task 1: X to A Frequency ---
snp_x_to_a = snp_df[
    (snp_df["HGVSc"].str.contains("[CGT]>A", na=False)) &
    (snp_df["HGVSc"].str.extract(r'c\.(\d+)[A-Z]>[A-Z]').astype(float)[0] <= 1332)
]
snp_total = len(snp_df)
snp_x_to_a_count = len(snp_x_to_a)
snp_x_to_a_frequency = snp_x_to_a_count / snp_total if snp_total > 0 else 0

delins_total = len(delins_df)
delins_x_to_a_count = 0
delins_x_to_a_frequency = 0

print("\n--- X to A Frequency ---")
print(f"SNP X to A mutations: {snp_x_to_a_count}/{snp_total}")
print(f"SNP Frequency: {snp_x_to_a_frequency:.4f} ({snp_x_to_a_frequency*100:.2f}%)")
print(f"DEL/INS X to A mutations: {delins_x_to_a_count}/{delins_total}")
print(f"DEL/INS Frequency: {delins_x_to_a_frequency:.4f} ({delins_x_to_a_frequency*100:.2f}%)")

# --- Task 2: X to A Mutations Creating AA/AAA/AXA at Mutation Site ---
snp_x_to_a_aa_axa_count = 0
snp_x_to_a_total = 0
for hgvsc in snp_x_to_a["HGVSc"]:
    if pd.notna(hgvsc):
        try:
            hgvsc_part = hgvsc.split(":")[1] if ":" in hgvsc else hgvsc
            pos_match = re.match(r'c\.(\d+)[A-Z]>[A-Z]', hgvsc_part)
            if not pos_match:
                continue
            pos = int(pos_match.group(1))
            pos_idx = pos - 1  # 0-based index
            # Check a 3-nt window around the mutation: pos-1 to pos+1
            start_idx = max(0, pos_idx - 1)  # Ensure we don't go below 0
            end_idx = min(len(mRNA), pos_idx + 2)  # Ensure we don't go beyond sequence length
            # Reference sequence around mutation
            ref_seq = mRNA[start_idx:end_idx]
            # Mutated sequence (replace position pos with A)
            mutated_seq = mRNA[start_idx:pos_idx] + 'a' + mRNA[pos_idx+1:end_idx]
            if snp_x_to_a_total < 5:  # Debug print for first 5
                print(f"SNP X to A HGVSc: {hgvsc}, Pos: {pos}, Ref: {ref_seq}, Mutated: {mutated_seq}")
            # Check additional windows for edge cases
            # pos-2 to pos
            if pos_idx - 2 >= 0:
                ref_seq_left = mRNA[pos_idx-2:pos_idx+1]
                mutated_seq_left = mRNA[pos_idx-2:pos_idx] + 'a'
            else:
                ref_seq_left = ref_seq
                mutated_seq_left = mutated_seq
            # pos to pos+2
            if pos_idx + 3 <= len(mRNA):
                ref_seq_right = mRNA[pos_idx:pos_idx+3]
                mutated_seq_right = 'a' + mRNA[pos_idx+1:pos_idx+3]
            else:
                ref_seq_right = ref_seq
                mutated_seq_right = mutated_seq
            # Check if the mutation creates AA/AAA/AXA
            if (has_aa_or_axa(mutated_seq) and not has_aa_or_axa(ref_seq)) or \
               (has_aa_or_axa(mutated_seq_left) and not has_aa_or_axa(ref_seq_left)) or \
               (has_aa_or_axa(mutated_seq_right) and not has_aa_or_axa(ref_seq_right)):
                snp_x_to_a_aa_axa_count += 1
            snp_x_to_a_total += 1
        except Exception as e:
            print(f"Error parsing SNP HGVSc {hgvsc}: {e}")
            continue

snp_x_to_a_aa_axa_frequency = snp_x_to_a_aa_axa_count / snp_total if snp_total > 0 else 0
delins_x_to_a_aa_axa_count = 0
delins_x_to_a_aa_axa_frequency = 0

print("\n--- X to A Mutations Creating AA/AAA/AXA at Mutation Site ---")
print(f"SNP X to A with AA/AAA/AXA at site: {snp_x_to_a_aa_axa_count}/{snp_total}")
print(f"SNP Frequency: {snp_x_to_a_aa_axa_frequency:.4f} ({snp_x_to_a_aa_axa_frequency*100:.2f}%)")
print(f"DEL/INS X to A with AA/AAA/AXA at site: {delins_x_to_a_aa_axa_count}/{delins_total}")
print(f"DEL/INS Frequency: {delins_x_to_a_aa_axa_frequency:.4f} ({delins_x_to_a_aa_axa_frequency*100:.2f}%)")

# Debug for DEL/INS (for the next section)
delins_debug_count = 0
for hgvsc in delins_df["HGVSc"]:
    if pd.notna(hgvsc) and "c." in hgvsc:
        try:
            hgvsc_part = hgvsc.split(":")[1] if ":" in hgvsc else hgvsc
            if "delins" in hgvsc_part or "del" in hgvsc_part or "dup" in hgvsc_part:
                end_match = re.match(r'c\.(\d+)_(\d+)(delins|del|dup)', hgvsc_part)
                if not end_match:
                    continue
                three_prime = int(end_match.group(2))
            elif "ins" in hgvsc_part:
                ins_match = re.match(r'c\.(\d+)_(\d+)ins', hgvsc_part)
                if not ins_match:
                    continue
                three_prime = int(ins_match.group(2))
            else:
                continue
            start_idx = three_prime
            if three_prime + 24 <= 1332:
                downstream = mRNA[start_idx:start_idx+24]
                if delins_debug_count < 5:
                    print(f"DEL/INS HGVSc: {hgvsc}, End: {three_prime}, Downstream: {downstream}")
                    delins_debug_count += 1
        except Exception as e:
            print(f"Error parsing DEL/INS HGVSc {hgvsc}: {e}")
            continue

# --- Task 3: Mutations with Single A, AA, AXA Downstream ---
snp_single_a_count = 0
snp_aa_count = 0
snp_axa_count = 0
snp_total_aa_axa = 0
for hgvsc in snp_df["HGVSc"]:
    if pd.notna(hgvsc) and ">" in hgvsc:
        try:
            hgvsc_part = hgvsc.split(":")[1] if ":" in hgvsc else hgvsc
            pos_match = re.match(r'c\.(\d+)[A-Z]>[A-Z]', hgvsc_part)
            if not pos_match:
                continue
            pos = int(pos_match.group(1))
            start_idx = pos
            if pos + 24 <= 1332:
                downstream = mRNA[start_idx:start_idx+24]
                if snp_total_aa_axa < 5:
                    print(f"SNP HGVSc: {hgvsc}, Pos: {pos}, Downstream: {downstream}")
                if has_aa(downstream):
                    snp_aa_count += 1
                elif has_axa(downstream):
                    snp_axa_count += 1
                elif has_single_a(downstream):
                    snp_single_a_count += 1
                snp_total_aa_axa += 1
        except Exception as e:
            print(f"Error parsing SNP HGVSc {hgvsc}: {e}")
            continue

snp_single_a_frequency = snp_single_a_count / snp_total if snp_total > 0 else 0
snp_aa_frequency = snp_aa_count / snp_total if snp_total > 0 else 0
snp_axa_frequency = snp_axa_count / snp_total if snp_total > 0 else 0
snp_a_aa_axa_frequency = (snp_single_a_count + snp_aa_count + snp_axa_count) / snp_total if snp_total > 0 else 0

delins_single_a_count = 0
delins_aa_count = 0
delins_axa_count = 0
delins_total_aa_axa = 0
for hgvsc in delins_df["HGVSc"]:
    if pd.notna(hgvsc) and "c." in hgvsc:
        try:
            hgvsc_part = hgvsc.split(":")[1] if ":" in hgvsc else hgvsc
            if "delins" in hgvsc_part or "del" in hgvsc_part or "dup" in hgvsc_part:
                end_match = re.match(r'c\.(\d+)_(\d+)(delins|del|dup)', hgvsc_part)
                if not end_match:
                    continue
                three_prime = int(end_match.group(2))
            elif "ins" in hgvsc_part:
                ins_match = re.match(r'c\.(\d+)_(\d+)ins', hgvsc_part)
                if not ins_match:
                    continue
                three_prime = int(ins_match.group(2))
            else:
                continue
            start_idx = three_prime
            if three_prime + 24 <= 1332:
                downstream = mRNA[start_idx:start_idx+24]
                if has_aa(downstream):
                    delins_aa_count += 1
                elif has_axa(downstream):
                    delins_axa_count += 1
                elif has_single_a(downstream):
                    delins_single_a_count += 1
                delins_total_aa_axa += 1
        except Exception as e:
            print(f"Error parsing DEL/INS HGVSc {hgvsc}: {e}")
            continue

delins_single_a_frequency = delins_single_a_count / delins_total if delins_total > 0 else 0
delins_aa_frequency = delins_aa_count / delins_total if delins_total > 0 else 0
delins_axa_frequency = delins_axa_count / delins_total if delins_total > 0 else 0
delins_a_aa_axa_frequency = (delins_single_a_count + delins_aa_count + delins_axa_count) / delins_total if delins_total > 0 else 0

total_mutations = snp_total + delins_total
total_single_a = snp_single_a_count + delins_single_a_count
total_aa = snp_aa_count + delins_aa_count
total_axa = snp_axa_count + delins_axa_count
total_a_aa_axa = total_single_a + total_aa + total_axa
total_single_a_frequency = total_single_a / total_mutations if total_mutations > 0 else 0
total_aa_frequency = total_aa / total_mutations if total_mutations > 0 else 0
total_axa_frequency = total_axa / total_mutations if total_mutations > 0 else 0
total_a_aa_axa_frequency = total_a_aa_axa / total_mutations if total_mutations > 0 else 0

print("\n--- Mutations with Single A, AA, AXA Downstream ---")
print(f"SNP Single A: {snp_single_a_count}/{snp_total}")
print(f"SNP Single A Frequency: {snp_single_a_frequency:.4f} ({snp_single_a_frequency*100:.2f}%)")
print(f"SNP AA: {snp_aa_count}/{snp_total}")
print(f"SNP AA Frequency: {snp_aa_frequency:.4f} ({snp_aa_frequency*100:.2f}%)")
print(f"SNP AXA (no AA): {snp_axa_count}/{snp_total}")
print(f"SNP AXA Frequency: {snp_axa_frequency:.4f} ({snp_axa_frequency*100:.2f}%)")
print(f"SNP A/AA/AXA: {snp_single_a_count + snp_aa_count + snp_axa_count}/{snp_total}")
print(f"SNP A/AA/AXA Frequency: {snp_a_aa_axa_frequency:.4f} ({snp_a_aa_axa_frequency*100:.2f}%)")
print(f"DEL/INS Single A: {delins_single_a_count}/{delins_total}")
print(f"DEL/INS Single A Frequency: {delins_single_a_frequency:.4f} ({delins_single_a_frequency*100:.2f}%)")
print(f"DEL/INS AA: {delins_aa_count}/{delins_total}")
print(f"DEL/INS AA Frequency: {delins_aa_frequency:.4f} ({delins_aa_frequency*100:.2f}%)")
print(f"DEL/INS AXA (no AA): {delins_axa_count}/{delins_total}")
print(f"DEL/INS AXA Frequency: {delins_axa_frequency:.4f} ({delins_axa_frequency*100:.2f}%)")
print(f"DEL/INS A/AA/AXA: {delins_single_a_count + delins_aa_count + delins_axa_count}/{delins_total}")
print(f"DEL/INS A/AA/AXA Frequency: {delins_a_aa_axa_frequency:.4f} ({delins_a_aa_axa_frequency*100:.2f}%)")
print(f"Combined Single A: {total_single_a}/{total_mutations}")
print(f"Combined Single A Frequency: {total_single_a_frequency:.4f} ({total_single_a_frequency*100:.2f}%)")
print(f"Combined AA: {total_aa}/{total_mutations}")
print(f"Combined AA Frequency: {total_aa_frequency:.4f} ({total_aa_frequency*100:.2f}%)")
print(f"Combined AXA (no AA): {total_axa}/{total_mutations}")
print(f"Combined AXA Frequency: {total_axa_frequency:.4f} ({total_axa_frequency*100:.2f}%)")
print(f"Combined A/AA/AXA: {total_a_aa_axa}/{total_mutations}")
print(f"Combined A/AA/AXA Frequency: {total_a_aa_axa_frequency:.4f} ({total_a_aa_axa_frequency*100:.2f}%)")

# --- Task 4: Overlap Between X to A Creating AA/AAA/AXA and AA/AXA Downstream ---
snp_overlap_aa_axa_count = 0
for hgvsc in snp_x_to_a["HGVSc"]:
    if pd.notna(hgvsc):
        try:
            hgvsc_part = hgvsc.split(":")[1] if ":" in hgvsc else hgvsc
            pos_match = re.match(r'c\.(\d+)[A-Z]>[A-Z]', hgvsc_part)
            if not pos_match:
                continue
            pos = int(pos_match.group(1))
            pos_idx = pos - 1
            # Check Category A condition
            start_idx = max(0, pos_idx - 1)
            end_idx = min(len(mRNA), pos_idx + 2)
            ref_seq = mRNA[start_idx:end_idx]
            mutated_seq = mRNA[start_idx:pos_idx] + 'a' + mRNA[pos_idx+1:end_idx]
            if pos_idx - 2 >= 0:
                ref_seq_left = mRNA[pos_idx-2:pos_idx+1]
                mutated_seq_left = mRNA[pos_idx-2:pos_idx] + 'a'
            else:
                ref_seq_left = ref_seq
                mutated_seq_left = mutated_seq
            if pos_idx + 3 <= len(mRNA):
                ref_seq_right = mRNA[pos_idx:pos_idx+3]
                mutated_seq_right = 'a' + mRNA[pos_idx+1:pos_idx+3]
            else:
                ref_seq_right = ref_seq
                mutated_seq_right = mutated_seq
            in_category_a = (has_aa_or_axa(mutated_seq) and not has_aa_or_axa(ref_seq)) or \
                            (has_aa_or_axa(mutated_seq_left) and not has_aa_or_axa(ref_seq_left)) or \
                            (has_aa_or_axa(mutated_seq_right) and not has_aa_or_axa(ref_seq_right))
            # Check Category B condition (AA/AXA in downstream)
            if in_category_a and pos + 24 <= 1332:
                downstream = mRNA[pos:pos+24]
                if has_aa_or_axa(downstream):
                    snp_overlap_aa_axa_count += 1
        except Exception as e:
            print(f"Error parsing SNP HGVSc for overlap {hgvsc}: {e}")
            continue

delins_overlap_aa_axa_count = 0  # DEL/INS has no X to A mutations
total_overlap_aa_axa = snp_overlap_aa_axa_count + delins_overlap_aa_axa_count

print("\n--- Overlap: X to A Creating AA/AAA/AXA Also with AA/AXA Downstream ---")
print(f"SNP Overlap: {snp_overlap_aa_axa_count}/{snp_x_to_a_aa_axa_count}")
print(f"DEL/INS Overlap: {delins_overlap_aa_axa_count}/{delins_x_to_a_aa_axa_count}")
print(f"Total Overlap: {total_overlap_aa_axa}/{snp_x_to_a_aa_axa_count}")

# --- Task 5: Overlap Between X to A and A/AA/AXA Downstream ---
snp_overlap_x_to_a_and_a_aa_axa = 0
for hgvsc in snp_x_to_a["HGVSc"]:
    if pd.notna(hgvsc):
        try:
            hgvsc_part = hgvsc.split(":")[1] if ":" in hgvsc else hgvsc
            pos_match = re.match(r'c\.(\d+)[A-Z]>[A-Z]', hgvsc_part)
            if not pos_match:
                continue
            pos = int(pos_match.group(1))
            if pos + 24 <= 1332:
                downstream = mRNA[pos:pos+24]
                if has_any_a(downstream):
                    snp_overlap_x_to_a_and_a_aa_axa += 1
        except Exception as e:
            print(f"Error parsing SNP HGVSc for X to A overlap {hgvsc}: {e}")
            continue

delins_overlap_x_to_a_and_a_aa_axa = 0  # DEL/INS has no X to A mutations
total_overlap_x_to_a_and_a_aa_axa = snp_overlap_x_to_a_and_a_aa_axa + delins_overlap_x_to_a_and_a_aa_axa

print("\n--- Overlap: X to A Also with A/AA/AXA Downstream ---")
print(f"SNP Overlap: {snp_overlap_x_to_a_and_a_aa_axa}/{snp_x_to_a_count}")
print(f"DEL/INS Overlap: {delins_overlap_x_to_a_and_a_aa_axa}/{delins_x_to_a_count}")
print(f"Total Overlap: {total_overlap_x_to_a_and_a_aa_axa}/{snp_x_to_a_count}")

# --- Task 6: Combined Frequency (X to A OR Any with A/AA/AXA, Overlap Subtracted) ---
snp_union = snp_x_to_a_count + (snp_single_a_count + snp_aa_count + snp_axa_count) - snp_overlap_x_to_a_and_a_aa_axa
snp_union_frequency = snp_union / snp_total if snp_total > 0 else 0

delins_union = delins_x_to_a_count + (delins_single_a_count + delins_aa_count + delins_axa_count) - delins_overlap_x_to_a_and_a_aa_axa
delins_union_frequency = delins_union / delins_total if delins_total > 0 else 0

total_union = snp_union + delins_union
total_union_frequency = total_union / total_mutations if total_mutations > 0 else 0

print("\n--- Combined Frequency (X to A OR Any with A/AA/AXA, Overlap Subtracted) ---")
print(f"SNP Union: {snp_union}/{snp_total}")
print(f"SNP Frequency: {snp_union_frequency:.4f} ({snp_union_frequency*100:.2f}%)")
print(f"DEL/INS Union: {delins_union}/{delins_total}")
print(f"DEL/INS Frequency: {delins_union_frequency:.4f} ({delins_union_frequency*100:.2f}%)")
print(f"Combined Union: {total_union}/{total_mutations}")
print(f"Combined Frequency: {total_union_frequency:.4f} ({total_union_frequency*100:.2f}%)")

# --- Plotting the Data ---
categories = ['SNP', 'DEL/INS', 'Combined']
x_to_a_frequencies = [snp_x_to_a_frequency * 100, delins_x_to_a_frequency * 100, (snp_x_to_a_count + delins_x_to_a_count) / total_mutations * 100]
x_to_a_aa_axa_frequencies = [snp_x_to_a_aa_axa_frequency * 100, delins_x_to_a_aa_axa_frequency * 100, (snp_x_to_a_aa_axa_count + delins_x_to_a_aa_axa_count) / total_mutations * 100]
single_a_frequencies = [snp_single_a_frequency * 100, delins_single_a_frequency * 100, total_single_a_frequency * 100]
aa_frequencies = [snp_aa_frequency * 100, delins_aa_frequency * 100, total_aa_frequency * 100]
axa_frequencies = [snp_axa_frequency * 100, delins_axa_frequency * 100, total_axa_frequency * 100]
union_frequencies = [snp_union_frequency * 100, delins_union_frequency * 100, total_union_frequency * 100]

# Bar Plot
x = np.arange(len(categories))
width = 0.15  # Adjusted width to fit more bars

fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width*2, x_to_a_frequencies, width, label='X to A', color='skyblue')
bars2 = ax.bar(x - width, x_to_a_aa_axa_frequencies, width, label='X to A Creating AA/AAA/AXA', color='lightgreen')
bars3 = ax.bar(x, single_a_frequencies, width, label='Single A Downstream', color='lightyellow')
bars4 = ax.bar(x + width, aa_frequencies, width, label='AA Downstream', color='salmon')
bars5 = ax.bar(x + width*2, axa_frequencies, width, label='AXA Downstream (no AA)', color='lightcoral')
bars6 = ax.bar(x + width*3, union_frequencies, width, label='Union (X to A OR A/AA/AXA)', color='violet')

ax.set_ylabel('Frequency (%)')
ax.set_title('Mutation Frequencies in TP53 (ENST00000269305.4)')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.set_ylim(0, 100)

for bars in [bars1, bars2, bars3, bars4, bars5, bars6]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/Users/jingkunzeng/Downloads/tp53_mutation_frequencies_bar.png')
plt.close()

# Pie Charts for Single A, AA, AXA
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
# SNP
snp_labels = ['Single A', 'AA', 'AXA (no AA)', 'Neither']
snp_values = [snp_single_a_frequency * 100, snp_aa_frequency * 100, snp_axa_frequency * 100,
              (1 - snp_single_a_frequency - snp_aa_frequency - snp_axa_frequency) * 100]
ax1.pie(snp_values, labels=snp_labels, autopct='%1.1f%%', colors=['lightyellow', 'salmon', 'lightcoral', 'lightgrey'])
ax1.set_title('SNP Downstream Motifs')
# DEL/INS
delins_labels = ['Single A', 'AA', 'AXA (no AA)', 'Neither']
delins_values = [delins_single_a_frequency * 100, delins_aa_frequency * 100, delins_axa_frequency * 100,
                 (1 - delins_single_a_frequency - delins_aa_frequency - delins_axa_frequency) * 100]
ax2.pie(delins_values, labels=delins_labels, autopct='%1.1f%%', colors=['lightyellow', 'salmon', 'lightcoral', 'lightgrey'])
ax2.set_title('DEL/INS Downstream Motifs')
# Combined
combined_labels = ['Single A', 'AA', 'AXA (no AA)', 'Neither']
combined_values = [total_single_a_frequency * 100, total_aa_frequency * 100, total_axa_frequency * 100,
                   (1 - total_single_a_frequency - total_aa_frequency - total_axa_frequency) * 100]
ax3.pie(combined_values, labels=combined_labels, autopct='%1.1f%%', colors=['lightyellow', 'salmon', 'lightcoral', 'lightgrey'])
ax3.set_title('Combined Downstream Motifs')

plt.tight_layout()
plt.savefig('/Users/jingkunzeng/Downloads/tp53_downstream_motifs_pie.png')
plt.close()

print("\nPlots saved as 'tp53_mutation_frequencies_bar.png' and 'tp53_downstream_motifs_pie.png' in Downloads.")
