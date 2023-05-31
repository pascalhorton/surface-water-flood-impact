import math


def compute_confusion_matrix(df, field_obs='target', field_model='predict'):
    """
    Compute the confusion matrix.

    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe containing the observations and the model predictions.
    field_obs: str
        The name of the field containing the observations.
    field_model
        The name of the field containing the model predictions.

    Returns
    -------
    The confusion matrix components (tp, tn, fp, fn).

    """
    tp = len(df[(df[field_obs] == 1) & (df[field_model] == 1)])
    tn = len(df[(df[field_obs] == 0) & (df[field_model] == 0)])
    fp = len(df[(df[field_obs] == 0) & (df[field_model] == 1)])
    fn = len(df[(df[field_obs] == 1) & (df[field_model] == 0)])

    return tp, tn, fp, fn


def compute_score(metric, tp, tn, fp, fn):
    """
    Metrics based on the confusion matrix.

    Parameters
    ----------
    metric: str
        The desired metric
    tp: int
        Cases where obs is true and model predicts occurrence
    tn: int
        Cases where obs is false and model does not predict occurrence
    fp: int
        Cases where obs is false and model predicts occurrence
    fn: int
        Cases where obs is true and model does not predict occurrence

    Returns
    -------
    The value of the desired metric.

    Authors
    ------
    Daniel Bernet: original R version, 2017
    Pascal Horton: conversion to Python, 2023
    """

    # Sample size
    n = tp + fp + fn + tn
    assert n > 0

    if metric in ['base_rate', 'brate']:
        # Base rate (brate) = (a+c) / n
        # [0, 1]
        brate = (tp + fn) / n
        assert (0 <= brate <= 1)
        return brate

    elif metric in ['forecast_rate', 'frate']:
        # Forcast rate (frate) = (a+b) / n
        # [0, 1]
        frate = (tp + fp) / n
        assert (0 <= frate <= 1)
        return frate

    elif metric in ['bias']:
        # Frequency Bias (bias) = (a+b) / (a+c)
        # [0, inf]
        bias = (tp + fp) / (tp + fn)
        assert bias >= 0
        return bias

    elif metric in ['hit_rate', 'H']:
        # Hit rate (H) = a / (a+c)
        # [0, 1]
        h = tp / (tp + fn)
        assert (0 <= h <= 1)
        return h

    elif metric in ['false_alarm_rate', 'F']:
        # False alarm rate (F) = b / (b+d)
        # Note f != false alarm ratio = b / (a+b)
        # [0, 1]
        f = fp / (fp + tn)
        assert (0 <= f <= 1)
        return f
    
    elif metric in ['false_alarm_ratio', 'FAR']:
        # False alarm ratio (FAR) = b / (a+b)
        # [0, 1]
        far = fp / (tp + fp)
        assert (0 <= far <= 1)
        return far
    
    elif metric in ['proportion_correct', 'PC']:
        # Proportion Correct (PC) = (a+d) / n
        # [0, 1]
        pc = (tp + tn) / n
        assert (0 <= pc <= 1)
        return pc

    elif metric in ['success_index', 'SI']:
        # Success index (si) = 1/2 (a/(a+c) + d/(b+d))
        # [0, 1]
        si = 1 / 2 * (tp / (tp + fn) +
                      tn / (fp + tn))
        assert (0 <= si <= 1)
        return si

    elif metric in ['critical_success_index', 'CSI']:
        # Critical Sucess Index (CSI) = Thread score = a / (a+b+c)
        # [0, 1]
        csi = tp / (tp + fp + fn)
        assert (0 <= csi <= 1)
        return csi

    elif metric in ['gilbert_skill_score', 'GSS']:
        # Gilbert Skill Score (GSS) = (a-a_exp) / (a+b+c-a_exp)
        # [-1/3, 1]
        # Expected a_obs1_mod1 for random forecast a_ex) = (a+b) * (a+c) / n
        a_exp = (tp + fp) * (tp + fn) / n
        gss = (tp - a_exp) / (tp + fp + fn - a_exp)
        assert ((-1 / 3) <= gss <= 1)
        return gss

    elif metric in ['heidke_skill_score', 'HSS']:
        # Heidke Skill Score (HSS) = (a+d-a_exp-d_exp) / (n-a_exp-d_exp)
        # [-1, 1]
        # Expected d_obs0_mod0 for random forecast a_exp = (b+d) * (c+d) / n
        a_exp = (tp + fp) * (tp + fn) / n
        d_exp = (fp + tn) * (fn + tn) / n
        hss = (tp + tn - a_exp - d_exp) / (n - a_exp - d_exp)
        assert (-1 <= hss <= 1)
        return hss

    elif metric in ['peirce_skill_score', 'PSS']:
        # Peirce Skill Score (PSS) = (ad-bc) / ((b+d)(a+c)) = H - F
        # [-1, 1]
        h = tp / (tp + fn)
        f = fp / (fp + tn)
        pss = h - f
        assert (-1 <= pss <= 1)
        return pss

    elif metric in ['clayton_skill_score', 'CSS']:
        # Clayton Skill Score (css) = a/(a+b) - c/(c+d)
        # [-1, 1]
        css = tp / (tp + fp) - fn / (
                fn + tn)
        assert (-1 <= css <= 1)
        return css

    elif metric in ['doolittle_skill_score', 'DSS']:
        # Doolittle Skill Score (DSS) = (ad-bc) / sqrt((a+b)(c+d)(a+c)(b+d))
        dss = (tp * tn - fp * fn) / math.sqrt(
            (tp + fp) * (fn + tn) *
            (tp + fn) * (fp + tn))
        assert (-1 <= dss <= 1)
        return dss

    elif metric in ['odds_ratio', 'OR']:
        # Odds Ratio (OR) = ad / bc
        # [0, Inf]
        odr = tp * tn / (fp * fn)
        assert (odr >= 0)
        return odr

    elif metric in ['log_odds_ratio', 'LOR']:
        # Log of Odds Ratio (LOR) = ln(ad / bc)
        # [-Inf, Inf]
        odr = tp * tn / (fp * fn)
        lodr = math.log(odr)
        return lodr

    elif metric in ['odds_ratio_skill_score', 'ORSS']:
        # Odds Ratio Skill Score (ORSS) = Yule's Q = (ad-bc) / (ad+bc) =
        # (or - 1) / (or + 1) = (H - F) / (H(1-F) + F(1-H))
        # [-1, 1]
        odr = tp * tn / (fp * fn)
        if math.isinf(odr):  # orss has no skill when odr = Inf
            return 0

        orss = (odr - 1) / (odr + 1)  # (h - f) / (h * (1 - f) + f * (1 - h))
        assert (-1 <= orss <= 1)
        return orss

    elif metric in ['extremal_dependency_score', 'EDS']:
        # Extremal Dependency Score (EDS) = 2 ln((a+c) / n) / ln(a / n) - 1
        eds = 2 * math.log((tp + fn) / n) / math.log(
            tp / n) - 1
        assert (-1 <= eds <= 1)
        return eds

    elif metric in ['symmetric_extreme_dependency_score', 'SEDS']:
        # Symmetric Extreme Dependency Score (SEDS) = ln(a_exp / a) / ln(a / n)
        if tp == 0:  # seds has no skill when a = 0
            return 0

        a_exp = (tp + fp) * (tp + fn) / n
        seds = math.log(a_exp / tp) / math.log(tp / n)
        assert (-1 <= seds <= 1)
        return seds

    elif metric in ['extremal_dependency_index', 'EDI']:
        # Extremal Dependence Index (EDI) = (ln(F)-ln(H)) / (ln(F)+ln(H))
        h = tp / (tp + fn)
        f = fp / (fp + tn)

        if h == 1 and f == 0:  # edi is undefined for perfect forecasts
            return 1
        elif h == 0:  # edi has no skill when h = 0
            return 0
        elif f == 0:  # edi is undefined when f = 0
            return math.nan

        edi = (math.log(f) - math.log(h)) / (math.log(f) + math.log(h))
        assert (-1 <= edi <= 1)
        return edi

    elif metric in ['symmetric_extremal_dependency_index', 'SEDI']:
        # Symmetric Extremal Dependence Index
        # (SEDI) = (ln(F)-ln(H)+ln(1-H)-ln(1-F)) /
        #          (ln(F)+ln(H)+ln(1-H)+ln(1-F))
        h = tp / (tp + fn)
        f = fp / (fp + tn)

        if h == 1 and f == 0:  # sedi is undefined for perfect forecasts
            return 1
        elif h == 0:
            return 0
        elif h == 1 or f == 0 or f == 1:
            return math.nan

        sedi = (math.log(f) - math.log(h) + math.log(1 - h) - math.log(1 - f)) / (
                math.log(f) + math.log(h) + math.log(1 - h) + math.log(1 - f))
        assert (-1 <= sedi <= 1)
        return sedi
