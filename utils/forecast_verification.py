import math


def binary_detection(metric, a_obs1_mod1, b_obs0_mod1, c_obs1_mod0, d_obs0_mod0):
    """
    Metrics based on the confusion matrix.

    Parameters
    ----------
    metric: str
        The desired metric
    a_obs1_mod1
        Cases where obs is true and model predicts occurrence
    b_obs0_mod1
        Cases where obs is false and model predicts occurrence
    c_obs1_mod0
        Cases where obs is true and model does not predict occurrence
    d_obs0_mod0
        Cases where obs is false and model does not predicts occurrence

    Returns
    -------
    The value of the desired score.

    Author
    ------
    Daniel Bernet (original R version, 2017)
    """

    # Sample size (n) = a + b + c + d
    n = a_obs1_mod1 + b_obs0_mod1 + c_obs1_mod0 + d_obs0_mod0
    assert (n > 0)

    if metric in ['base_rate', 'brate']:
        # Base rate (brate) = (a+c) / n
        # [0,1]
        brate = (a_obs1_mod1 + c_obs1_mod0) / n
        assert (brate >= 0 & brate <= 1)
        return brate

    elif metric in ['forecast_rate', 'frate']:
        # Forcast rate (frate) = (a+b) / n
        # [0,1]
        frate = (a_obs1_mod1 + b_obs0_mod1) / n
        assert (frate >= 0 & frate <= 1)
        return frate

    elif metric in ['bias']:
        # Frequency Bias (bias) = (a+b) / (a+c)
        # [0,inf]
        bias = (a_obs1_mod1 + b_obs0_mod1) / (a_obs1_mod1 + c_obs1_mod0)
        assert bias >= 0
        return bias

    elif metric in ['hit_rate', 'H']:
        # Hit rate (H) = a / (a+c)
        # [0,1]
        h = a_obs1_mod1 / (a_obs1_mod1 + c_obs1_mod0)
        assert (h >= 0 & h <= 1)
        return h

    elif metric in ['false_alarm_rate', 'F']:
        # False alarm rate (F) = b / (b+d)
        # Note f != false alarm ratio = b / (a+b)
        # [0,1]
        f = b_obs0_mod1 / (b_obs0_mod1 + d_obs0_mod0)
        assert (f >= 0 & f <= 1)
        return f
    
    elif metric in ['false_alarm_ratio', 'FAR']:
        # False alarm ratio (FAR) = b / (a+b)
        # [0,1]
        far = b_obs0_mod1 / (a_obs1_mod1 + b_obs0_mod1)
        assert (far >= 0 & far <= 1)
        return far
    
    elif metric in ['proportion_correct', 'PC']:
        # Proportion Correct (PC) = (a+d) / n
        # [0,1]
        pc = (a_obs1_mod1 + d_obs0_mod0) / n
        assert (pc >= 0 & pc <= 1)
        return pc

    elif metric in ['success_index', 'SI']:
        # Success index (si) = 1/2 (a/(a+c) + d/(b+d))
        # [0,1]
        si = 1 / 2 * (a_obs1_mod1 / (a_obs1_mod1 + c_obs1_mod0) + 
                      d_obs0_mod0 / (b_obs0_mod1 + d_obs0_mod0))
        assert (si >= 0 & si <= 1)
        return si

    elif metric in ['critical_success_index', 'CSI']:
        # Critical Sucess Index (CSI) = Thread score = a / (a+b+c)
        # [0,1]
        csi = (a_obs1_mod1) / (a_obs1_mod1 + b_obs0_mod1 + c_obs1_mod0)
        assert (csi >= 0 & csi <= 1)
        return csi

    elif metric in ['gilbert_skill_score', 'GSS']:
        # Gilbert Skill Score (GSS) = (a-a_exp) / (a+b+c-a_exp)
        # [-1/3,1]
        # Expected a_obs1_mod1 for random forecast a_ex) = (a+b) * (a+c) / n
        a_exp = (a_obs1_mod1 + b_obs0_mod1) * (a_obs1_mod1 + c_obs1_mod0) / n
        gss = (a_obs1_mod1 - a_exp) / (a_obs1_mod1 + b_obs0_mod1 + c_obs1_mod0 - a_exp)
        assert (gss >= (-1 / 3) & gss <= 1)
        return gss

    elif metric in ['heidke_skill_score', 'HSS']:
        # Heidke Skill Score (HSS) = (a+d-a_exp-d_exp) / (n-a_exp-d_exp)
        # [-1,1]
        # Expected d_obs0_mod0 for random forecast a_exp = (b+d) * (c+d) / n
        a_exp = (a_obs1_mod1 + b_obs0_mod1) * (a_obs1_mod1 + c_obs1_mod0) / n
        d_exp = (b_obs0_mod1 + d_obs0_mod0) * (c_obs1_mod0 + d_obs0_mod0) / n
        hss = (a_obs1_mod1 + d_obs0_mod0 - a_exp - d_exp) / (n - a_exp - d_exp)
        assert (hss >= -1 & hss <= 1)
        return hss

    elif metric in ['peirce_skill_score', 'PSS']:
        # Peirce Skill Score (PSS) = (ad-bc) / ((b+d)(a+c)) = H - F
        # [-1,1]
        h = a_obs1_mod1 / (a_obs1_mod1 + c_obs1_mod0)
        f = b_obs0_mod1 / (b_obs0_mod1 + d_obs0_mod0)
        pss = h - f
        assert (pss >= -1 & pss <= 1)
        return pss

    elif metric in ['clayton_skill_score', 'CSS']:
        # Clayton Skill Score (css) = a/(a+b) - c/(c+d)
        # [-1,1]
        css = a_obs1_mod1 / (a_obs1_mod1 + b_obs0_mod1) - c_obs1_mod0 / (
                c_obs1_mod0 + d_obs0_mod0)
        assert (css >= -1 & css <= 1)
        return css

    elif metric in ['doolittle_skill_score', 'DSS']:
        # Doolittle Skill Score (DSS) = (ad-bc) / sqrt((a+b)(c+d)(a+c)(b+d))
        dss = (a_obs1_mod1 * d_obs0_mod0 - b_obs0_mod1 * c_obs1_mod0) / math.sqrt(
            (a_obs1_mod1 + b_obs0_mod1) * (c_obs1_mod0 + d_obs0_mod0) *
            (a_obs1_mod1 + c_obs1_mod0) * (b_obs0_mod1 + d_obs0_mod0))
        assert (dss >= -1 & dss <= 1)
        return dss

    elif metric in ['odds_ratio', 'OR']:
        # Odds Ratio (OR) = ad / bc
        # [0,Inf]
        odr = a_obs1_mod1 * d_obs0_mod0 / (b_obs0_mod1 * c_obs1_mod0)
        assert (odr >= 0)
        return odr

    elif metric in ['log_odds_ratio', 'LOR']:
        # Log of Odds Ratio (LOR) = ln(ad / bc)
        # [-Inf,Inf]
        odr = a_obs1_mod1 * d_obs0_mod0 / (b_obs0_mod1 * c_obs1_mod0)
        lodr = math.log(odr)
        return lodr

    elif metric in ['odds_ratio_skill_score', 'ORSS']:
        # Odds Ratio Skill Score (ORSS) = Yule's Q = (ad-bc) / (ad+bc) = (or - 1) / (or + 1) = (H - F) / (H(1-F) + F(1-H))
        # [-1,1]
        odr = a_obs1_mod1 * d_obs0_mod0 / (b_obs0_mod1 * c_obs1_mod0)
        if math.isinf(odr):  # orss has no skill when odr = Inf
            return 0

        orss = (odr - 1) / (odr + 1)  # (h - f) / (h * (1 - f) + f * (1 - h))
        assert (orss >= -1 & orss <= 1)
        return orss

    elif metric in ['extremal_dependency_score', 'EDS']:
        # Extremal Dependency Score (EDS) = 2 ln((a+c) / n) / ln(a / n) - 1
        eds = 2 * math.log((a_obs1_mod1 + c_obs1_mod0) / n) / math.log(
            a_obs1_mod1 / n) - 1
        assert (eds >= -1 & eds <= 1)
        return eds

    elif metric in ['symmetric_extreme_dependency_score', 'SEDS']:
        # Symmetric Extreme Dependency Score (SEDS) = ln(a_exp / a) / ln(a / n)
        if a_obs1_mod1 == 0:  # seds has no skill when a = 0
            return 0

        a_exp = (a_obs1_mod1 + b_obs0_mod1) * (a_obs1_mod1 + c_obs1_mod0) / n
        seds = math.log(a_exp / a_obs1_mod1) / math.log(a_obs1_mod1 / n)
        assert (seds >= -1 & seds <= 1)
        return seds

    elif metric in ['extremal_dependency_index', 'EDI']:
        # Extremal Dependence Index (EDI) = (ln(F)-ln(H)) / (ln(F)+ln(H))
        h = a_obs1_mod1 / (a_obs1_mod1 + c_obs1_mod0)
        f = b_obs0_mod1 / (b_obs0_mod1 + d_obs0_mod0)

        if h == 1 & f == 0:  # edi is undefined for perfect forecasts
            return 1
        elif h == 0:  # edi has no skill when h = 0
            return 0
        elif f == 0:  # edi is undefined when f = 0
            return math.nan

        edi = (math.log(f) - math.log(h)) / (math.log(f) + math.log(h))
        assert (edi >= -1 & edi <= 1)
        return edi

    elif metric in ['symmetric_extremal_dependency_index', 'SEDI']:
        # Symmetric Extremal Dependence Index (SEDI) = (ln(F)-ln(H)+ln(1-H)-ln(1-F)) /
        #                                              (ln(F)+ln(H)+ln(1-H)+ln(1-F))
        h = a_obs1_mod1 / (a_obs1_mod1 + c_obs1_mod0)
        f = b_obs0_mod1 / (b_obs0_mod1 + d_obs0_mod0)

        if h == 1 & f == 0:  # sedi is undefined for perfect forecasts
            return 1
        elif h == 0:
            return 0
        elif h == 1 | f == 0 | f == 1:
            return math.nan

        sedi = (math.log(f) - math.log(h) + math.log(1 - h) - math.log(1 - f)) / (
                math.log(f) + math.log(h) + math.log(1 - h) + math.log(1 - f))
        assert (sedi >= -1 & sedi <= 1)
        return sedi
