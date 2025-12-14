from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from data import load_dataset
from feat_extraction import get_features
from feat_selection import pca_selector, fisher_selector, filter_features
from laplacian_score import dtw_laplacian_score, lb_dtw_laplacian_score, euclidean_laplacian_score


class Experiment:
    def __init__(self, clf, dataset: str, denoise: bool = True, use_y: bool = False, k: int = 3, t: float = 1.0, **kwargs):
        self.clf = clf
        self.dataset = dataset
        self.desc = ""

        self.k = k
        self.t = t

        self.denoise = denoise
        self.pre_dtw_filename = ""
        if not denoise:
            self.desc += "raw "
            self.pre_dtw_filename = "raw_"

        self.desc += dataset

        self.use_y = use_y
        self.kwargs = kwargs

        if use_y:
            self.desc += " with y"
        self.desc += f"; {k = }, {t = }; {type(clf).__name__}"


    def extract_scores(self, X, X_f, y):
        self.scores = {}

        self.scores["pca"] = pca_selector(X_f)
        self.scores["fisher"] = fisher_selector(X_f, y)

        y = y if self.use_y else None

        self.scores["euclidean"] = euclidean_laplacian_score(X_f, self.k, self.t, y)
        self.scores["dtw"] = dtw_laplacian_score(X, X_f, self.k, self.t, y, f"dtw/{self.pre_dtw_filename}dtw_{self.dataset}_train.npy")
        self.scores["lb_dtw"] = lb_dtw_laplacian_score(X, X_f, self.k, self.t, y, f"dtw/{self.pre_dtw_filename}lb_dtw_{self.dataset}_train.npy")

        print("Scores extracted.")

    def run(self, r_list: list[int]):
        self.X, self.y = load_dataset(self.dataset, self.denoise, **self.kwargs)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        X_train_f = get_features(X_train, **self.kwargs)
        X_test_f = get_features(X_test, **self.kwargs)

        scaler = StandardScaler()
        X_train_f = scaler.fit_transform(X_train_f)
        X_test_f = scaler.transform(X_test_f)

        metrics = {}

        self.clf.fit(X_train_f, y_train)
        metrics["all_feats"] = self.clf.score(X_test_f, y_test)

        self.extract_scores(X_train, X_train_f, y_train)

        for filter_method, scores in self.scores.items():
            metrics[filter_method] = []

            for r in r_list:
                X_sel, feat_idx = filter_features(X_train_f, scores, r, return_feat_idx=True)
                X_test_sel = X_test_f[:, feat_idx]

                self.clf.fit(X_sel, y_train)
                metrics[filter_method].append(self.clf.score(X_test_sel, y_test))

        return metrics


def plot_experiment(ax, exp: dict):
    desc, r_list, metrics = exp["desc"], exp["r_list"], exp["metrics"]

    all_feats = metrics["all_feats"]

    pca = metrics["pca"]
    fisher = metrics["fisher"]
    euclidean = metrics["euclidean"]
    dtw = metrics["dtw"]
    lb_dtw = metrics["lb_dtw"]

    ax.plot(r_list, pca, label="PCA")
    ax.plot(r_list, fisher, label="Fisher")
    ax.plot(r_list, euclidean, label="Euclidean LS")
    ax.plot(r_list, dtw, label="DTW LS")
    ax.plot(r_list, lb_dtw, label="Lower-bound DTW LS")

    ax.hlines(all_feats, xmin=0, xmax=max(r_list), label="No selection", linestyle='--', color="black")

    ax.set_title(desc)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of Features (r)")
    ax.set_ylabel("Accuracy")
    ax.legend()

    ax.grid(True, linestyle=':', alpha=0.6)


def plot_experiments(experiments: list[dict]):
    n_exps = len(experiments)
    shift = n_exps // 4

    for i in range(n_exps // 2):
        exp_raw = experiments[i]
        exp_denoised = experiments[i + shift]

        _, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

        plot_experiment(axes[0], exp_raw)
        plot_experiment(axes[1], exp_denoised)

        plt.show()
