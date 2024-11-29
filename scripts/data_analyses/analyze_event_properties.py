import pandas as pd
import matplotlib.pyplot as plt

data_dir = R'C:\Data\Projects\2024 SWF\Data\CombiPrecip'
events_parquet = data_dir + '/events_cpc_model_domain_3x3_2005_2024.parquet'
percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

threshold_sum = 150
threshold_duration = 150
threshold_mean_intensity = 15
threshold_max_intensity = 50


df = pd.read_parquet(events_parquet)

stats = pd.DataFrame()
stats['duration'] = df['duration'].describe(percentiles=percentiles)
stats['p_sum'] = df['p_sum'].describe(percentiles=percentiles)
stats['i_mean'] = df['i_mean'].describe(percentiles=percentiles)
stats['i_max'] = df['i_max'].describe(percentiles=percentiles)
stats['api'] = df['api'].describe(percentiles=percentiles)

# Save the statistics to a CSV file
stats.to_csv(data_dir + '/event_statistics.csv')

# Print the statistics to the console
print(stats.to_string(float_format="{:.1f}".format))

# Plot duration, sum and max intensity
cmap = plt.get_cmap('viridis')
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
filtered_df = df[df['duration'] <= threshold_duration]
filtered_df['duration'].plot.hist(bins=50, title="Duration [h]", density=True, ax=ax[0], color=cmap(0.2))
ax[0].set_xlim(0, threshold_duration)
ax[0].set_ylabel("Density")

filtered_df = df[df['p_sum'] <= threshold_sum]
filtered_df['p_sum'].plot.hist(bins=50, title="Precipitation sum [mm]", density=True, ax=ax[1], color=cmap(0.5))
ax[1].set_xlim(0, threshold_sum)
ax[1].set_ylabel(None)

filtered_df = df[df['i_max'] <= threshold_max_intensity]
filtered_df['i_max'].plot.hist(bins=50, title="Maximum intensity [mm/h]", density=True, ax=ax[2], color=cmap(0.8))
ax[2].set_xlim(0, threshold_max_intensity)
ax[2].set_ylabel(None)

plt.tight_layout()
plt.savefig(data_dir + '/event_properties_histograms.png')
plt.savefig(data_dir + '/event_properties_histograms.pdf')

plt.show()
