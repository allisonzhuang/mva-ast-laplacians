from sklearn.model_selection import train_test_split

from data import load_dataset
from feat_extraction import get_features
from feat_selection import pca_selector, fisher_selector, filter_features
from laplacian_score import dtw_laplacian_score, lb_dtw_laplacian_score


class Experiment:
    def __init__(self, dataset: str, k: int, t: float, desc: str = "", **kwargs):
        self.dataset = dataset
        self.desc = desc

        self.k = k
        self.t = t

        self.kwargs = kwargs

    def extract_scores(self):
        self.scores = {}

        self.scores["pca"] = pca_selector(self.X_f)
        self.scores["fisher"] = fisher_selector(self.X_f, self.y)
        self.scores["dtw"] = dtw_laplacian_score(self.X, self.X_f, self.k, self.t, f"dtw/dtw_{self.dataset}.npy")
        self.scores["lb_dtw"] = lb_dtw_laplacian_score(self.X, self.X_f, self.k, self.t, f"dtw/lb_dtw_{self.dataset}.npy")

        print("Scores extracted.")

    def run(self, r_list: list[int], clf):
        self.X, self.y = load_dataset(self.dataset, **self.kwargs)
        self.X_f = get_features(self.X, **self.kwargs)

        metrics = {}

        X_train, X_test, y_train, y_test = train_test_split(self.X_f, self.y, test_size=0.2, random_state=42, stratify=self.y)

        clf.fit(X_train, y_train)
        metrics["all_feats"] = clf.score(X_test, y_test)

        self.extract_scores()

        for filter_method, scores in self.scores.items():
            metrics[filter_method] = []

            for r in r_list:
                X_sel, feat_idx = filter_features(X_train, scores, r, return_feat_idx=True)

                clf.fit(X_sel, y_train)
                metrics[filter_method].append(clf.score(X_test[:, feat_idx], y_test))

        return metrics
