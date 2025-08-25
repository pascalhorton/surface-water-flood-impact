"""
Test script for loading and evaluating different pre-trained models.
"""
import pandas as pd

from swafi.config import Config
from swafi.impact_basic_options import ImpactBasicOptions
from swafi.impact_thr import ImpactThresholds
from swafi.precip_combiprecip import CombiPrecip
from swafi.utils.verification import compute_confusion_matrix, print_classic_scores, \
    assess_roc_auc, store_classic_scores

config = Config()


def main():
    options = ImpactBasicOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()

    # Extract precipitation events
    cpc = CombiPrecip(config.get('YEAR_START_TEST'), config.get('YEAR_END_TEST'))
    cpc.open_files(config.get('DIR_PRECIP'))
    cpc.apply_smoothing(filter_size=3)
    events = cpc.extract_events()



    # Create the impact function
    thr = ImpactThresholds(options)

    # Restrict the features to the ones used in the 2019 method
    thr.tabular_features = {'event': ['i_max_q', 'p_sum_q']}
    thr.load_features(['event'])

    print(f"Threshold 2019 method (union):")
    thr.set_thresholds(thr_i_max=0.9, thr_p_sum=0.98, method='union')


    y_pred = thr.model.predict(x)
    df_res = pd.DataFrame(columns=['split'])
    df_tmp = pd.DataFrame(columns=df_res.columns)
    df_tmp['split'] = ['test']

    tp, tn, fp, fn = compute_confusion_matrix(y, y_pred)
    print_classic_scores(tp, tn, fp, fn)
    store_classic_scores(tp, tn, fp, fn, df_tmp)
    y_pred_prob = thr.model.predict_proba(x)
    roc = assess_roc_auc(y, y_pred_prob[:, 1])
    df_tmp['ROC_AUC'] = [roc]

    print(f"Threshold 2019 method (intersection):")
    thr.set_thresholds(thr_i_max=0.9, thr_p_sum=0.98, method='intersection')
    thr.assess_model_on_all_periods(save_results=True, file_tag='thr2019_intersect')


if __name__ == '__main__':
    main()
