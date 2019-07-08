import numpy as np
import os
import sklearn.metrics as sk
import torch


def cal_metrics(measure_in, measure_out, in_out_lbl):
    measure_all = np.concatenate([measure_in, measure_out])
    auroc = sk.roc_auc_score(in_out_lbl, measure_all) # AUROC
    aupr_in = sk.average_precision_score(in_out_lbl, measure_all) # aupr in
    aupr_out = sk.average_precision_score((in_out_lbl - 1) * -1, measure_all * -1) # aupr out
    out_mea_mean = np.mean(measure_out) # Mean of out-dist measure
    return auroc, aupr_in, aupr_out, out_mea_mean


def create_ood_lbl(measure_in, measure_out):
    all_mea = [np.ones_like(measure_in), np.zeros_like(measure_out)]
    return np.concatenate(all_mea, axis=0)


# PT version
def get_predictions(model, datasets, keys):
    preds = {}
    for name in sorted(datasets.keys()):
        print("Getting predictions {}".format(name))
        outputs = model.predict_np(datasets[name], 128)
        pred = dict(zip(keys, outputs))
        preds[name] = pred
    return preds


# PT version
def ood_detection_eval(preds, id_dataset, measurement, ood_datasets):
    mea_id = preds[id_dataset][measurement]
    max_mea_id = np.max(mea_id, axis=1)
    print('Below OOD detections are done by {}'.format(measurement))
    print('-----------------------------------------')
    print('|  auroc  | aupr_in | auprout | avg_mea |')
    print('-----------------------------------------')
    if ood_datasets is None:
        ood_datasets = sorted([i for i in preds.keys() if i != id_dataset])
    
    for d in ood_datasets:
        mea_ood = preds[d][measurement]
        max_mea_ood = np.max(mea_ood, axis=1)
        ood_lbl = create_ood_lbl(max_mea_id, max_mea_ood)
        auroc, aupr_in, aupr_out, out_mea_mean = cal_metrics(
            max_mea_id, max_mea_ood, ood_lbl
        )
        print('| {:.5f} | {:.5f} | {:.5f} | {:.5f} | {}'.format(
            auroc, aupr_in, aupr_out, out_mea_mean, d
        ))

    print('-----------------------------------------')
