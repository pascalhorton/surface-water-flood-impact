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


def _plot_importance_if_available(model, run_name, dir_output):
    """Plot the feature importance if the model exposes it (RF, LightGBM)."""
    plot = getattr(model, 'plot_feature_importance', None)
    if plot is None:
        return
    try:
        plot(tag='feature_importance_' + run_name, dir_output=dir_output)
    except Exception as exc:
        logger.warning("Could not plot the feature importance: %s", exc)


def save_best_model(options, events, study, setup_model, model_kind, dir_output):
    """
    Refit the best trial of the study and save the resulting (split) model.

    The model is fit on the chronological training split and assessed on the
    held-out validation split (the honest performance estimate), with its
    decision threshold tuned on that validation split.

    The best trial only stores the tuned hyperparameters; the fixed ones
    (n_estimators, weight_denominator, ...) are taken from the base options.
    With the fixed random state and the deterministic (chronological) split,
    refitting reproduces the trial's model exactly. The decision threshold was
    not tuned during the search (the objectives are threshold-free / at a fixed
    threshold), so it is tuned here on the validation split.

    Applicable to the tabular models (RF, LightGBM), i.e. those exposing
    ``fit()``, ``tune_probability_threshold()`` and ``save_model()``. It only
    ever saves the split model; a deployment model refit on the whole period is
    left to the caller (see ``refit_and_save_full_period``), as it is only sound
    for a model whose fit() does not early-stop on the validation split.

    Parameters
    ----------
    options : ImpactBasicOptions
        The base options, carrying the fixed (non-searched) hyperparameters.
    events : Events
        The events object.
    study : optuna.study.Study
        The completed study.
    setup_model : callable(options, events) -> Impact
        Builds the model (split, class weights, features): the train script's
        ``_setup_model``.
    model_kind : str
        Short model tag used in the file names and result tags ('rf', 'lgbm').
    dir_output : str
        The directory where to save the model, results and plots.

    Returns
    -------
    tuple(Impact, float, str)
        The fitted model (with its splits still populated), the tuned decision
        threshold, and the base name used for the saved file. These let the
        caller refit and save further models (e.g. the full-period one).
    """
    best_trial = study.best_trial
    logger.info("Refitting the best model (trial %s, value %.5f) to save it.",
                best_trial.number, best_trial.value)

    options_best = options.copy()
    # The trial params share the option names, so apply them by name; the
    # hyperparameters kept out of the search retain their base values.
    for key, value in best_trial.params.items():
        setattr(options_best, key, value)
    # Behave as a plain fit from here on (show the resolved params, not the
    # Optuna search space).
    options_best.optimize_with_optuna = False
    options_best.print_options(show_optuna_params=True)

    base_name = (f'model_{model_kind}_{options_best.dataset}_'
                 f'{options_best.event_method}_{options_best.precip_dataset}')

    # Fit on the training split, assess and tune the threshold on the held-out
    # validation split.
    model = setup_model(options_best, events)
    model.fit()
    threshold = model.tune_probability_threshold()
    logger.info("Optimal probability threshold (tuned on validation): %.4f",
                threshold)
    model.assess_model_on_all_periods(
        save_results=True, file_tag=f'{model_kind}_{options_best.run_name}')
    _plot_importance_if_available(model, options_best.run_name, dir_output)
    model.save_model(dir_output=dir_output, base_name=base_name)
    logger.info("Best (split) model saved in %s", dir_output)

    return model, threshold, base_name