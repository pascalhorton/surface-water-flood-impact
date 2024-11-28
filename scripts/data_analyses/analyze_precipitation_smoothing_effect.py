import pandas as pd

data_dir = R'\\mobistorage02.giub.unibe.ch\share\SFW2023\02_data\CPC\precip_events_20241127'
cpc_orig = data_dir + '/events_cpc_model_domain_nos_2005_2024.parquet'
cpc_smoothed = data_dir + '/events_cpc_model_domain_3x3_2005_2024.parquet'


def compare_duration(df):
    diff = df['duration_smoothed'] - df['duration_orig']
    pc_diff = 100 * diff / df['duration_orig']
    pc_diff = pc_diff.dropna()
    print(f"Comparison of duration:")
    print(f"  - Mean of original dataset: {df['duration_orig'].mean():.4f}")
    print(f"  - Mean of smoothed dataset: {df['duration_smoothed'].mean():.4f}")
    print(f"  - Mean difference: {diff.mean():.4f}")
    print(f"  - Mean relative difference: {pc_diff.mean():.4f}%")
    print()


df_orig = pd.read_parquet(cpc_orig)
df_smoothed = pd.read_parquet(cpc_smoothed)

# Show some basic statistics on the duration
print("Statistics on the duration:")
print(f"  - Original dataset: {df_orig['duration'].describe()}")
print(f"  - Smoothed dataset: {df_smoothed['duration'].describe()}")

# Add a column with the event ID (eid)
df_orig['eid'] = df_orig.index
df_smoothed['eid'] = df_smoothed.index

l_orig = len(df_orig)
l_smoothed = len(df_smoothed)

print(f"Length of original dataset: {l_orig}")
print(f"Length of smoothed dataset: {l_smoothed}")

# Select the events that are common to both datasets (based on the cid, e_start, e_end)
df_orig = df_orig.set_index(['cid', 'e_start', 'e_end'])
df_smoothed = df_smoothed.set_index(['cid', 'e_start', 'e_end'])
df_common = df_orig.join(df_smoothed, how='inner', lsuffix='_orig', rsuffix='_smoothed')
pc_common = 100 * len(df_common) / l_orig
print(f"Length of common dataset: {len(df_common)} ({pc_common:.4f}%)")

# Remove the common events from the original datasets (using the eid)
df_orig = df_orig.reset_index()
df_smoothed = df_smoothed.reset_index()
df_orig = df_orig[~df_orig['eid'].isin(df_common['eid_orig'])]
df_smoothed = df_smoothed[~df_smoothed['eid'].isin(df_common['eid_smoothed'])]

# Select the events that are have the same cid and e_start, but different e_end
df_orig = df_orig.set_index(['cid', 'e_start'])
df_smoothed = df_smoothed.set_index(['cid', 'e_start'])
df_common_start = df_orig.join(df_smoothed, how='inner', lsuffix='_orig', rsuffix='_smoothed')
df_orig = df_orig.reset_index()
df_smoothed = df_smoothed.reset_index()
df_orig = df_orig[~df_orig['eid'].isin(df_common_start['eid_orig'])]
df_smoothed = df_smoothed[~df_smoothed['eid'].isin(df_common_start['eid_smoothed'])]
pc_common_start = 100 * len(df_common_start) / l_orig
print(f"Events with common start: {len(df_common_start)} ({pc_common_start:.4f}%)")
compare_duration(df_common_start)

# Select the events that are have the same cid and e_end, but different e_start
df_orig = df_orig.set_index(['cid', 'e_end'])
df_smoothed = df_smoothed.set_index(['cid', 'e_end'])
df_common_end = df_orig.join(df_smoothed, how='inner', lsuffix='_orig', rsuffix='_smoothed')
df_orig = df_orig.reset_index()
df_smoothed = df_smoothed.reset_index()
df_orig = df_orig[~df_orig['eid'].isin(df_common_end['eid_orig'])]
df_smoothed = df_smoothed[~df_smoothed['eid'].isin(df_common_end['eid_smoothed'])]
pc_common_end = 100 * len(df_common_end) / l_orig
print(f"Events with common end: {len(df_common_end)} ({pc_common_end:.4f}%)")
compare_duration(df_common_end)

# Select the events that are have the same cid, but still overlap
df_orig = df_orig.reset_index()
df_smoothed = df_smoothed.reset_index()
df_merged = df_orig.merge(df_smoothed, on='cid', suffixes=('_orig', '_smoothed'))
df_overlap = df_merged[(df_merged['e_start_orig'] < df_merged['e_end_smoothed']) &
                       (df_merged['e_end_orig'] > df_merged['e_start_smoothed'])]
df_orig = df_orig[~df_orig['eid'].isin(df_overlap['eid_orig'])]
df_smoothed = df_smoothed[~df_smoothed['eid'].isin(df_overlap['eid_smoothed'])]
pc_overlap = 100 * len(df_overlap) / l_orig
print(f"Events with overlap: {len(df_overlap)} ({pc_overlap:.4f}%)")
compare_duration(df_overlap)

# Show the statistics of the remaining events
print(f"Events remaining in the original dataset: {len(df_orig)} "
      f"({100 * len(df_orig) / l_orig:.4f}%)")


# For the common events, compare the precipitation values (p_sum, i_mean, i_max, api)
print("Comparison for the common events:")
for col in ['p_sum', 'i_mean', 'i_max', 'api', 'p_sum_q', 'i_mean_q', 'i_max_q', 'api_q']:
    col_orig = col + '_orig'
    col_smoothed = col + '_smoothed'
    diff = df_common[col_orig] - df_common[col_smoothed]
    pc_diff = 100 * diff / df_common[col_orig]
    pc_diff = pc_diff.dropna()
    print(f"Comparison of {col}:")
    print(f"  - Mean of original dataset: {df_common[col_orig].mean():.4f}")
    print(f"  - Mean of smoothed dataset: {df_common[col_smoothed].mean():.4f}")
    print(f"  - Max of original dataset: {df_common[col_orig].max():.4f}")
    print(f"  - Max of smoothed dataset: {df_common[col_smoothed].max():.4f}")
    print(f"  - Mean difference: {diff.mean():.4f}")
    print(f"  - Mean relative difference: {pc_diff.mean():.4f}%")
    print()

print(f"Done.")


