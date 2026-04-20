import logging

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass


logger = logging.getLogger(__name__)


def get_or_create_optuna_study(options):
    """
    Get or create an Optuna study.

    Parameters
    ----------
    options: ImpactBasicOptions
        The options.

    Returns
    -------
    optuna.study.Study
        The Optuna study.
    """
    if not has_optuna:
        raise ValueError("Optuna is not installed")

    file_path = f"./{options.optuna_study_name}.log"
    lock_obj = optuna.storages.journal.JournalFileOpenLock(file_path)  # For Windows
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(file_path, lock_obj=lock_obj)
    )

    sampler = None
    if options.optuna_random_sampler:
        sampler = optuna.samplers.RandomSampler()

    try:
        study = optuna.load_study(
            study_name=options.optuna_study_name,
            storage=storage,
            sampler=sampler
        )
        logger.info("Study '%s' already exists.", options.optuna_study_name)
    except KeyError:
        # If the study does not exist, create it
        study = optuna.create_study(
            study_name=options.optuna_study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler
        )
        logger.info("Study '%s' created successfully.", options.optuna_study_name)

    return study