from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

subdir = 'mobiliar_occurrence_all'
files_dir = Path(Rf'D:\Projects\2023 SWF\Analyses\Skill models\{subdir}')

files = [f for f in files_dir.glob('*.csv')]

# Load them all into a single DataFrame
dfs = []
for f in files:
    df = pd.read_csv(f)
    if '_thr2019_intersect_' in f.name:
        df['model'] = 'thr2019 intersect'
    elif '_thr2019_union_' in f.name:
        df['model'] = 'thr2019 union'
    elif '_lr_' in f.name:
        df['model'] = 'LR'
    elif '_rf_' in f.name:
        df['model'] = 'RF'
    elif '_tx_' in f.name:
        df['model'] = 'Transformer'
    else:
        df['model'] = 'unknown'
    dfs.append(df)

df = pd.concat(dfs)

# Define the desired order of the models
model_order = ['thr2019 intersect', 'thr2019 union', 'LR', 'RF', 'Transformer']
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

    # Validation
    valid_data = [
        df.loc[(df['split'] == 'valid') & (df['model'] == model), score].dropna() for
        model in model_order]
    ax[1].boxplot(valid_data, tick_labels=model_order)
    ax[1].set_title('Validation')

    # Test
    test_data = [
        df.loc[(df['split'] == 'test') & (df['model'] == model), score].dropna() for
        model in model_order]
    ax[2].boxplot(test_data, tick_labels=model_order)
    ax[2].set_title('Test')

    plt.suptitle(score)
    plt.tight_layout()
    plt.show()



