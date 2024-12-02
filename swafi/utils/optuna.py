
has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass


def get_or_create_optuna_study(options, random_sampler=False):
    if not has_optuna:
        raise ValueError("Optuna is not installed")

    file_path = f"./{options.optuna_study_name}.log"
    lock_obj = optuna.storages.journal.JournalFileOpenLock(file_path)  # For Windows
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(file_path, lock_obj=lock_obj)
    )

    sampler = None
    if random_sampler:
        sampler = optuna.samplers.RandomSampler()

    try:
        study = optuna.load_study(
            study_name=options.optuna_study_name,
            storage=storage,
            sampler=sampler
        )
        print(f"Study '{options.optuna_study_name}' already exists.")
    except KeyError:
        # If the study does not exist, create it
        study = optuna.create_study(
            study_name=options.optuna_study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler
        )
        print(f"Study '{options.optuna_study_name}' created successfully.")

    return study