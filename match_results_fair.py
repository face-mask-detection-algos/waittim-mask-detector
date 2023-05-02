import pandas as pd
import numpy as np
import argparse
from scipy import stats
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--ground_truth', type=str, required=True)
    parser.add_argument('--output', type=str, default="fairness_metrics.csv")
    return parser.parse_args()

def get_stats(df_sub, df_complete, column, invert_prop):
    stats_sub = df_sub.groupby(column).count().iloc[:,0]
    stats_complete = df_complete.groupby(column).count().iloc[:,0]
    if invert_prop:
        stats_sub = stats_complete - stats_sub
    prop = stats_sub / stats_complete
    
    num_sub = stats_sub.sum()
    num_total = stats_complete.sum()
    num_other = num_total - stats_complete
    num_other_sub = num_sub - stats_sub
    prop_other = num_other_sub / num_other
    
    df_stats = pd.concat([stats_complete, prop, num_other, prop_other], axis=1)
    df_stats.columns = ["n_1", "p_1", "n_2", "p_2"]
    df_stats["pooled_p"] = (df_stats.n_1*df_stats.p_1 + df_stats.n_2*df_stats.p_2)/(df_stats.n_1 + df_stats.n_2)
    df_stats["test_statistic"] = (df_stats.p_1 - df_stats.p_2)/np.sqrt(df_stats.pooled_p*(1-df_stats.pooled_p)*(1/df_stats.n_1 + 1/df_stats.n_2))
    # calculate p-value
    df_stats["p-value"] = 2 * (1 - stats.norm.cdf(np.abs(df_stats.test_statistic)))
    
    return df_stats

def main():
    args = get_args()
    final_columns = ["age", "gender", "race"]

    predictions = pd.read_csv(args.predictions, header=None, names=["path", "class", "x", "y", "w", "h"])
    predictions["name"] = predictions["path"].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    predictions["class"] = predictions["class"].astype(int).apply(lambda x: "nomask" if x == 0 else "mask")
    print(predictions.head())

    # clean up ground truth
    ground_truth = pd.read_csv(args.ground_truth)
    ground_truth = ground_truth.drop(ground_truth.columns[5:], axis=1)
    ground_truth["name"] = ground_truth["file"].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # get number of predictions per image and merge info in ground truth
    predictions_grouped = predictions.groupby("name")
    preds_per_image = predictions_grouped.count()[["path"]].rename(columns={"path": "num_preds"}).reset_index()
    ground_truth = pd.merge(ground_truth, preds_per_image, on="name", how="left")

    # get number of no mask
    preds_no_mask = predictions[predictions["class"] == "nomask"].groupby("name").count()[["path"]].rename(columns={"path": "num_no_mask"}).reset_index()
    ground_truth = pd.merge(ground_truth, preds_no_mask, on="name", how="left")

    # get number of mask
    preds_mask = predictions[predictions["class"] == "mask"].groupby("name").count()[["path"]].rename(columns={"path": "num_mask"}).reset_index()
    ground_truth = pd.merge(ground_truth, preds_mask, on="name", how="left")

    ground_truth = ground_truth.fillna(0)

    # NO RECOGNITION - num_mask == 0 and num_no_mask == 0
    no_recognition = ground_truth[(ground_truth["num_mask"] == 0) & (ground_truth["num_no_mask"] == 0)]
    rec_props = {attr: get_stats(no_recognition, ground_truth, attr, invert_prop=True) for attr in final_columns}

    # FALSE POSITIVE - num_mask > 0 and num_no_mask == 0
    false_positive = ground_truth[(ground_truth["num_mask"] > 0) & (ground_truth["num_no_mask"] == 0)]
    true_pos_props = {attr: get_stats(false_positive, ground_truth, attr, invert_prop=True) for attr in final_columns}

    # CORRECT PREDICTIONS - num_no_mask > 0
    # correct_predictions = ground_truth[ground_truth["num_no_mask"] > 0]
    # correct_preds_props = {attr: get_proportions(correct_predictions, ground_truth, attr) for attr in final_columns}

    for attr in final_columns:
        rec_props[attr].to_csv(args.output.replace(".csv", f"_{attr}_localization.csv"))
        true_pos_props[attr].to_csv(args.output.replace(".csv", f"_{attr}_false_positive.csv"))


if __name__ == "__main__":
    main()