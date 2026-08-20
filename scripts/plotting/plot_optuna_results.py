import optuna
import pandas as pd

# Location of the database file
study_name = "rf_04"
filepath = Rf'C:\Data\Projects\2024 SWF\Analyses\04 Random forest v2\02 Optuna runs\Mobiliar\Classic events\{study_name}.log'

# Load the study from the file
lock_obj = optuna.storages.journal.JournalFileOpenLock(filepath)  # For Windows
storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(filepath, lock_obj=lock_obj)
)

study = optuna.load_study(
    study_name=study_name, storage=storage
)

# Filter for completed trials with a non-None value
sorted_trials = [t for t in study.trials if
                 t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]

# Sort by value
sorted_trials.sort(key=lambda t: t.value, reverse=True)

for i, trial in enumerate(sorted_trials[:3]):
    print(f"Trial {i + 1}:")
    print("  Id: ", trial.number)
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


# Get the top 5 trials
top_trials = sorted_trials[:5]

# Collect all unique parameter names
all_params = set()
for trial in top_trials:
    all_params.update(trial.params.keys())

# Build a dict for the DataFrame
data = {}
for i, trial in enumerate(top_trials, 1):
    data[f"{i}"] = [trial.params.get(param, None) for param in all_params]
    data[f"{i}"].append(trial.value)

# Add 'Parameter' as index
index = list(all_params) + ['value']

# Create and print the DataFrame
df = pd.DataFrame(data, index=index)
print(df)

fig = optuna.visualization.plot_optimization_history(study)
fig.show()

fig = optuna.visualization.plot_slice(study)
fig.show()

# Analyze the failing trials
failed_trials = [t for t in study.trials if t.value == 0.0]
print(f"Number of failed trials: {len(failed_trials)}")

fig = optuna.visualization.plot_param_importances(study)
fig.show()
