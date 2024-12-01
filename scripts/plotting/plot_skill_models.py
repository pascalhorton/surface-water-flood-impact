from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

subdir = 'GVZ'
#subdir = 'Mobiliar'
files_dir = Path(Rf'C:\Data\Projects\2024 SWF\Analyses\07 Final assessments\{subdir}')

files = [f for f in files_dir.rglob('*.csv')]

# Remove the files containing the options
files = [f for f in files if '_options.csv' not in f.name]

# Load them all into a single DataFrame
dfs = []
for f in files:
    df = pd.read_csv(f)
    if '_bench_false_' in f.name:
        df['model'] = 'benchmark false'
    elif '_bench_true_' in f.name:
        df['model'] = 'benchmark true'
    elif '_bench_rand_' in f.name:
        df['model'] = 'benchmark random'
    elif '_thr2019_intersect_' in f.name:
        df['model'] = 'thr2019 intersect'
    elif '_thr2019_union_' in f.name:
        df['model'] = 'thr2019 union'
    elif '_lr_event_atts_' in f.name:
        df['model'] = 'LR ev atts'
    elif '_lr_event_and_all_static_atts_' in f.name:
        df['model'] = 'LR all atts'
    elif '_lr_event_and_static_atts_' in f.name:
        df['model'] = 'LR std atts'
    elif '_rf_' in f.name:
        df['model'] = 'RF'
    elif '_tx_' in f.name:
        df['model'] = 'Transformer'
    else:
        df['model'] = 'unknown'
    dfs.append(df)

df = pd.concat(dfs)

# Define the desired order of the models
model_order = ['benchmark false', 'benchmark true', 'benchmark random',
               'thr2019 intersect', 'thr2019 union',
               'LR ev atts', 'LR all atts', 'LR std atts',
               'RF', 'Transformer']
df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)

scores = df.columns[2:-1]

for score in scores:
    print(f"{score}:")
    print(df.groupby('model')[score].describe())
    print()

    # Plot the different scores as boxplots
    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharey=True)

    # Training
    train_data = [
        df.loc[(df['split'] == 'train') & (df['model'] == model), score].dropna() for
        model in model_order]
    ax[0].boxplot(train_data, tick_labels=model_order)
    ax[0].set_title('Training')
    ax[0].set_xticklabels(model_order, rotation=45, ha='right')

    # Validation
    valid_data = [
        df.loc[(df['split'] == 'valid') & (df['model'] == model), score].dropna() for
        model in model_order]
    ax[1].boxplot(valid_data, tick_labels=model_order)
    ax[1].set_title('Validation')
    ax[1].set_xticklabels(model_order, rotation=45, ha='right')

    # Test
    test_data = [
        df.loc[(df['split'] == 'test') & (df['model'] == model), score].dropna() for
        model in model_order]
    ax[2].boxplot(test_data, tick_labels=model_order)
    ax[2].set_title('Test')
    ax[2].set_xticklabels(model_order, rotation=45, ha='right')

    plt.suptitle(score)
    plt.tight_layout()
    plt.show()



