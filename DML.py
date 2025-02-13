import torch
print(torch.cuda.is_available())

import numpy as np
import torch

from functools import reduce
from operator import mul
from sklearn.model_selection import train_test_split
from typing import Tuple


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gaussian(z, sigma=1.0):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z / (sigma ** 2))


def m0_g0_s1(x: np.ndarray, majority: np.ndarray, minority: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = x[majority, 1] + 10 * x[majority, 3] + 5 * x[majority, 6]
    m0[minority] = 10 * x[minority, 1] + x[minority, 3] + 5 * x[minority, 6]

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    g0[majority] = x[majority, 0] + 10 * x[majority, 2] + 5 * x[majority, 5]
    g0[minority] = 10 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 5]

    return m0, g0


def m0_g0_s2(x: np.ndarray, majority: np.ndarray, minority: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = (np.maximum(0, x[majority, 1] + 10 * x[majority, 3] + 5 * x[majority, 6])) ** (1 / 2)
    m0[minority] = (np.maximum(0, 10 * x[minority, 1] + x[minority, 3] + 5 * x[minority, 6])) ** (1 / 2)

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    g0[majority] = (np.maximum(0, x[majority, 0] + 10 * x[majority, 2] + 5 * x[majority, 5])) ** (1 / 2)
    g0[minority] = (np.maximum(0, 10 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 5])) ** (1 / 2)

    return m0, g0


def m0_g0_s3(x: np.ndarray, majority: np.ndarray, minority: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = x[majority, 1] + 100 * x[majority, 3] + 5 * x[majority, 6]
    m0[minority] = 100 * x[minority, 1] + x[minority, 3] + 5 * x[minority, 6]

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    g0[majority] = (np.maximum(0, x[majority, 0] + 100 * x[majority, 2] + 5 * x[majority, 5])) ** (1 / 2)
    g0[minority] = (np.maximum(0, 100 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 5])) ** (1 / 2)

    return m0, g0


def m0_g0_s4(x: np.ndarray, majority: np.ndarray, minority: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # w_majority = 1 / np.linspace(1.0, 2.0, num=x.shape[1])
    # w_minority = 1 / np.logspace(1.0, 2.0, num=x.shape[1])
    #
    # m0 = np.zeros(x.shape[0], dtype=np.float64)
    # m0[majority] = gaussian(x[majority] @ w_majority)
    # m0[minority] = np.square(x[minority] @ w_minority)
    #
    # g0 = np.zeros(x.shape[0], dtype=np.float64)
    # g0[majority] = (np.maximum(0, x[majority, 0] + 100 * x[majority, 2] + 5 * x[majority, 5])) ** (1 / 2)
    # g0[minority] = (np.maximum(0, 100 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 5])) ** (1 / 2)

    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = (np.maximum(0, x[majority, 0] + 100 * x[majority, 2] + 5 * x[majority, 6])) ** (1 / 2)
    m0[minority] = (np.maximum(0, 100 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 6])) ** (1 / 2)

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    g0[majority] = (np.maximum(0, x[majority, 0] + 100 * x[majority, 2] + 5 * x[majority, 5])) ** (1 / 2)
    g0[minority] = (np.maximum(0, 100 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 5])) ** (1 / 2)

    return m0, g0


def m0_g0_s5(x: np.ndarray, majority: np.ndarray, minority: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    # g0[majority] = x[majority, -1] + np.absolute(x[majority, 2]) + 0.5 * np.exp(x[majority, 5] + x[majority, 4])
    # g0[minority] = x[minority, -1] + np.absolute(-1.0 * x[minority, 3]) - 2.5 * x[minority, 4]

    g0 = x[:, -1] + np.absolute(x[:, 2]) + 0.5 * np.exp(x[:, 5] + x[:, 4])

    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = np.maximum(0, 0.5 * x[majority, 1] ** 2 + x[majority, 5] + x[majority, 3] ** 3)
    m0[minority] = np.maximum(0, -2.5 * x[minority, 1] ** 2 + x[minority, -1] + x[minority, 4])

    return m0, g0


def m0_g0_s6(x: np.ndarray, majority: np.ndarray, minority: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = (np.maximum(0, x[majority, 0] + 100 * x[majority, 2] + 5 * x[majority, 6])) ** (1 / 2)
    m0[minority] = (np.maximum(0, 100 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 6])) ** (1 / 2)

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    g0[majority] = (np.maximum(0, x[majority, 1] + 100 * x[majority, 3] + 5 * x[majority, 5])) ** (1 / 2)
    g0[minority] = (np.maximum(0, 100 * x[minority, 1] + x[minority, 3] + 5 * x[minority, 5])) ** (1 / 2)

    return m0, g0


M0_G0_SETUPS = {
    "s1": m0_g0_s1,
    "s2": m0_g0_s2,
    "s3": m0_g0_s3,
    "s4": m0_g0_s4,
    "s5": m0_g0_s5,
    "s6": m0_g0_s6,
}


def m0_g0(x: np.ndarray, majority_s: float = 0.75, setup: str = "s1"):
    n_obs = x.shape[0]
    z = np.argsort(x[:, 0])
    threshold_idx = z[int(n_obs * majority_s)]
    threshold_val = x[threshold_idx, 0]
    majority = x[:, 0] < threshold_val
    minority = x[:, 0] >= threshold_val
    m0, g0 = M0_G0_SETUPS[setup](x, majority, minority)
    return m0, g0, (majority, minority)


class Data:

    def __train_test__(self, x, d, y, train_size: float = 0.5, seed: int = 42, **kwargs):
        x_train, x_test, d_train, d_test, y_train, y_test = train_test_split(
            x, d, y, train_size=train_size, random_state=seed, shuffle=True
        )

        train = {"x": x_train, "d": d_train, "y": y_train}
        test = {"x": x_test, "d": d_test, "y": y_test}

        return train, test


class DataSynthetic(Data):
    def __ar_covariance_params__(self, dim: int, ar_rho: float):
        mu = np.zeros(dim,)

        rho = np.ones(dim,) * ar_rho
        sigma = np.zeros(shape=(dim, dim))
        for i in range(dim):
            for j in range(i, dim):
                sigma[i][j] = reduce(mul, [rho[k] for k in range(i, j)], 1)
        sigma = np.triu(sigma) + np.triu(sigma).T - np.diag(np.diag(sigma))

        return mu, sigma

    def __init__(
        self,
        nb_features: int = 9,
        nb_observations: int = 90,
        ar_rho: float = 0.8,
        sigma_v: float = 1.0,
        sigma_u: float = 1.0,
        majority_s: float = 0.75,
        m0_g0_setup: str = "s1",
        as_tensors: bool = True,
    ):
        super().__init__()

        self._nb_observations = nb_observations
        self._sigma_v = sigma_v
        self._sigma_u = sigma_u
        self._majority_s = majority_s
        self._m0_g0_setup = m0_g0_setup
        self._as_tensors = as_tensors
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mu, self.sigma = self.__ar_covariance_params__(dim=nb_features, ar_rho=ar_rho)

    @classmethod
    def init_from_opts(cls, opts, as_tensors: bool = True):
        return cls(
            nb_features=opts.nb_features,
            nb_observations=opts.nb_observations,
            ar_rho=opts.ar_rho,
            sigma_v=opts.sigma_v,
            sigma_u=opts.sigma_u,
            majority_s=opts.majority_s,
            m0_g0_setup=opts.m0_g0_setup,
            as_tensors=as_tensors,
        )

    @property
    def nb_observations(self):
        return self._nb_observations

    @nb_observations.setter
    def nb_observations(self, new_nb_observations: int):
        self._nb_observations = new_nb_observations

    @property
    def sigma_v(self):
        return self._sigma_v

    @sigma_v.setter
    def sigma_v(self, new_sigma_v: float):
        self._sigma_v = new_sigma_v

    @property
    def sigma_u(self):
        return self._sigma_u

    @sigma_u.setter
    def sigma_u(self, new_sigma_u: float):
        self._sigma_u = new_sigma_u

    @property
    def majority_s(self):
        return self._majority_s

    @majority_s.setter
    def majority_s(self, new_majority_s: float):
        self._majority_s = new_majority_s

    @property
    def m0_g0_setup(self):
        return self._m0_g0_setup

    @m0_g0_setup.setter
    def m0_g0_setup(self, new_m0_g0_setup: str):
        self._m0_g0_setup = new_m0_g0_setup

    @property
    def as_tensors(self):
        return self._as_tensors

    @as_tensors.setter
    def as_tensors(self, new_as_tensors: bool):
        self._as_tensors = new_as_tensors

    def prep(self, real_theta: float):
        x = self.rng.multivariate_normal(self.mu, self.sigma, self.nb_observations)
        m_0, g_0, (majority, minority) = m0_g0(x, majority_s=self.majority_s, setup=self.m0_g0_setup)

        d = m_0 + self.sigma_v * self.rng.randn(self.nb_observations)
        y = d * real_theta + g_0 + self.sigma_u * self.rng.randn(self.nb_observations,)

        if self.as_tensors:
            x = torch.tensor(x).to(self._device)
            d = torch.tensor(d).to(self._device)
            y = torch.tensor(y).to(self._device)
            majority = torch.tensor(majority).to(self._device)
            minority = torch.tensor(minority).to(self._device)

        return x, d, y, (majority, minority)

    def generate(self, real_theta: float, train_size: float = 0.5, seed: int = 42, **kwargs):
        #####################################################################
        np.random.seed(seed)
        self.rng = np.random if seed is None else np.random.RandomState(seed)
        #####################################################################

        x, d, y, (majority, minority) = self.prep(real_theta)
        return self.train_test(x, d, y, majority, minority, train_size, seed)

    def train_test(self, x, d, y, majority, minority, train_size: float = 0.5, seed: int = 42, **kwargs):
        (
            x_train,
            x_test,
            d_train,
            d_test,
            y_train,
            y_test,
            majority_train,
            majority_test,
            minority_train,
            minority_test,
        ) = train_test_split(x, d, y, majority, minority, train_size=train_size, random_state=seed, shuffle=True)

        train = {"x": x_train, "d": d_train, "y": y_train, "majority": majority_train, "minority": minority_train}
        test = {"x": x_test, "d": d_test, "y": y_test, "majority": majority_test, "minority": minority_test}

        return train, test
DATA_TYPES = {"synthetic": DataSynthetic,}
import abc
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from argparse import Namespace
from pathlib import Path
from sklearn.base import BaseEstimator
from torch import Tensor
from typing import Any, Dict, Tuple

sns.set_style("darkgrid")
plt.rcParams["font.family"] = "serif"


class BaseModel(BaseEstimator):

    HISTORY_KEYS = ("train loss", "train theta hat", "test loss", "test theta hat")

    @staticmethod
    @abc.abstractmethod
    def name():
        raise NotImplementedError

    def __init__(self, **kwargs):
        super().__init__()

        self.params = kwargs
        self.history = None
        self.reset_history()

    def __str__(self):
        return self.__class__.__name

    @classmethod
    @abc.abstractmethod
    def init_from_opts(cls, opts: Namespace, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def fit_params(self, opts: Namespace, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def restart(self):
        raise NotImplementedError

    def reset_history(self):
        self.history = {k: [] for k in self.HISTORY_KEYS}

    @abc.abstractmethod
    def fit(self, train: Dict[str, Tensor], test: Dict[str, Tensor], **kwargs):
        """
        Fit the model to the data.
        :param train: a dictionary with train data.
        :param test: a dictionary with test data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predicts (m(x), l(x)) for a given input x.
        :param x: a tensor of shape (num_samples, num_features).
        :return: m(x) and l(x), each of shape (num_samples, ).
        """
        raise NotImplementedError

    @staticmethod
    def _normalize_inputs(
        train: Dict[str, Tensor], test: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        y_range = train["y"].max() - train["y"].min()
        train["d"] = train["d"] / y_range
        train["y"] = train["y"] / y_range
        test["d"] = test["d"] / y_range
        test["y"] = test["y"] / y_range

        return train, test

    def _arange_stat(self, key):
        return np.arange(start=0, stop=len(self.history[key]))

    def plot_history(self, real_theta: float, save_as: Path):

        hist_copy = copy.deepcopy(self.history)

        num_train_entries = sum(["train" in x for x in hist_copy.keys()])
        num_test_entries = sum(["test" in x for x in hist_copy.keys()])
        assert num_train_entries == num_test_entries

        min_test_loss_index = hist_copy["test loss"].index(min(hist_copy["test loss"]))
        test_theta_hat = hist_copy["test theta hat"][min_test_loss_index]  # [-1]

        _, axs = plt.subplots(2, num_train_entries, figsize=(7.5 * num_train_entries / 2, 7.5))

        axs[0, 0].plot(self._arange_stat("train loss"), hist_copy["train loss"])
        axs[0, 0].set_ylabel("loss")
        axs[0, 0].set_title("train loss")
        axs[0, 0].set_yscale("log")
        del hist_copy["train loss"]

        axs[0, 1].plot(self._arange_stat("train theta hat"), hist_copy["train theta hat"])
        axs[0, 1].set_ylabel("$\hat{\\theta}$")
        axs[0, 1].set_title("train theta estimation")
        axs[0, 1].axhline(y=real_theta, color="red", alpha=0.8, linestyle="--")
        del hist_copy["train theta hat"]

        train_keys = {k: v for k, v in hist_copy.items() if "train" in k}
        for i, x in enumerate(train_keys.items()):
            k, v = x
            axs[0, i + 2].plot(self._arange_stat(k), v)
            axs[0, i + 2].set_title(k + "/{:.3}".format(hist_copy[k][min_test_loss_index]))

        axs[1, 0].plot(self._arange_stat("test loss"), hist_copy["test loss"])
        axs[1, 0].set_ylabel("loss")
        axs[1, 0].set_title("test loss")
        axs[1, 0].set_yscale("log")
        axs[1, 0].axvline(x=min_test_loss_index, color="red", alpha=0.8, linestyle="--")
        del hist_copy["test loss"]

        axs[1, 1].plot(self._arange_stat("test theta hat"), self.history["test theta hat"])
        axs[1, 1].set_ylabel("$\hat{\\theta}$")
        axs[1, 1].set_title("test theta estimation")
        axs[1, 1].axvline(x=min_test_loss_index, color="red", alpha=0.8, linestyle="--")
        axs[1, 1].axhline(y=real_theta, color="red", alpha=0.8, linestyle="--")
        del hist_copy["test theta hat"]

        test_keys = {k: v for k, v in hist_copy.items() if "test" in k}
        for i, x in enumerate(test_keys.items()):
            k, v = x
            axs[1, i + 2].plot(self._arange_stat(k), v)
            axs[1, i + 2].set_title(k + "/{:.3}".format(hist_copy[k][min_test_loss_index]))
            axs[1, i + 2].axvline(x=min_test_loss_index, color="red", alpha=0.8, linestyle="--")

        plt.suptitle(f"theta = {real_theta}, theta estimation = {round(test_theta_hat, 3)}")
        plt.subplots_adjust(wspace=0.45, hspace=0.25)
        plt.savefig(save_as, bbox_inches="tight")
        plt.close()

import tempfile
import torch

from argparse import Namespace

from torch import Tensor
from typing import Any, Dict, Tuple




import numpy as np
import torch

from numpy import ndarray
from torch import Tensor
from typing import Tuple, Union


def est_theta_numpy(y: ndarray, d: ndarray, m_hat: ndarray, l_hat: ndarray) -> Tuple[float, float]:
    v_hat = d - m_hat
    mean_v_hat_2 = np.mean(v_hat * v_hat)
    theta_hat = np.mean(v_hat * (y - l_hat)) / mean_v_hat_2
    return theta_hat.item(), mean_v_hat_2.item()


def est_theta_torch(y: Tensor, d: Tensor, m_hat: Tensor, l_hat: Tensor) -> Tuple[float, float]:
    v_hat = d - m_hat
    mean_v_hat_2 = torch.mean(v_hat * v_hat)
    theta_hat = torch.mean(v_hat * (y - l_hat)) / mean_v_hat_2

    return theta_hat.item(), mean_v_hat_2.item()


def pearson_correlation_numpy(x: np.ndarray, y: np.ndarray):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    coeff = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return coeff


def pearson_correlation_torch(x: Tensor, y: Tensor):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    coeff = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return coeff.item()


def est_theta(
    y: Union[ndarray, Tensor], d: Union[ndarray, Tensor], m_hat: Union[ndarray, Tensor], l_hat: Union[ndarray, Tensor]
) -> Tuple[float, float]:
    if all(isinstance(i, Tensor) for i in (y, d, m_hat, l_hat)):
        return est_theta_torch(y, d, m_hat, l_hat)
    if all(isinstance(i, ndarray) for i in (y, d, m_hat, l_hat)):
        return est_theta_numpy(y, d, m_hat, l_hat)
    raise TypeError


def pearson_correlation(x: Union[ndarray, Tensor], y: Union[ndarray, Tensor]) -> Tuple[float, float]:
    if all(isinstance(i, Tensor) for i in (x, y)):
        return pearson_correlation_torch(x, y)
    if all(isinstance(i, ndarray) for i in (x, y)):
        return pearson_correlation_numpy(x, y)
    raise TypeError


import json
import numpy as np
import pandas as pd
import random
import string

from argparse import Namespace
from pathlib import Path


def gen_random_string(length: int):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def dump_opts_to_json(opts: Namespace, path: Path):
    params = vars(opts)
    with open(path, "w") as f:
        json.dump(params, f, indent=4)


def calc_cv_optimal_gamma_for_experiment(df: pd.DataFrame, thresh: float, sort_by: str = "y_res.2") -> float:
    df_x = df[~np.isnan(df["gamma"])]
    ref_df_sorted = df_x.sort_values(by=sort_by, ascending=True)
    optimal_gamma = ref_df_sorted.iloc[0]["gamma"].item()

    return optimal_gamma
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns

sns.set_style("darkgrid")
plt.rcParams["font.family"] = "serif"

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from typing import Any, Dict, Union


COLORS = (
    ("dimgray", "lightgray"),
    ("royalblue", "deepskyblue"),
    ("mediumvioletred", "palevioletred"),
    ("yellowgreen", "olivedrab"),
)
FACECOLORS = ("C0", "C2", "C4", "C6")


def plot_method_statistics(df: pd.DataFrame, n_exp: float, save_as: pathlib.Path):
    nb_columns = len(df.columns)
    _, axs = plt.subplots(1, nb_columns, figsize=(5 * nb_columns, 5))

    curr_axs = 0
    if "theta estimation" in df.columns:
        mean_theta_estimation = np.mean(df["theta estimation"].to_numpy())
        sns.histplot(data=df["theta estimation"], alpha=0.5, ax=axs[curr_axs], bins=20, kde=True)
        axs[curr_axs].set_title("$\hat{\\theta}$" + "\n" + "MEAN = {:.3f}".format(mean_theta_estimation))
        curr_axs += 1

    if "empirical bias" in df.columns:
        empirical_bias = df["empirical bias"].to_numpy()
        mean_bias = np.mean(empirical_bias)
        stderr = np.std(empirical_bias) / n_exp

        p = sns.violinplot(data=df["empirical bias"], ax=axs[curr_axs], orient="v")
        p.set_title(
            "empirical bias" + "($\\Delta \\theta$)" + "\n" + "MEAN = {:.3f} ± {:.3f}".format(mean_bias, stderr)
        )
        p.set_xlabel("")
        p.set_ylabel("")
        axs[curr_axs].axhline(y=0, linestyle="--", alpha=0.5, color="red")
        curr_axs += 1

    if "residuals correlation" in df.columns:
        mean_residuals_correlation = np.mean(df["residuals correlation"].to_numpy())

        p = sns.boxplot(data=df["residuals correlation"], ax=axs[curr_axs], orient="v")
        p.set_title("residuals correlation" + "\n" + "MEAN = {:.3f}".format(mean_residuals_correlation))
        p.set_xlabel("")
        p.set_ylabel("")
        hlines = set([round(i) for i in axs[curr_axs].get_ylim()])
        for hline in hlines:
            axs[curr_axs].axhline(y=hline, linestyle="--", alpha=0.5, color="red")
        curr_axs += 1

    if "res_m.2" in df.columns:
        mean_res_m_2 = np.mean(df["res_m.2"].to_numpy())

        p = sns.boxplot(data=df["res_m.2"], ax=axs[curr_axs], orient="v")
        p.set_title(r"$\left( D - \hat{m}_0 \right)^2$" + "\n" + "MEAN = {:.3e}".format(mean_res_m_2))
        p.set_xlabel("")
        p.set_ylabel("")
        p.set_ylim(bottom=0.0, top=np.percentile(df["res_m.2"].to_numpy(), 95))
        p.set_yscale("symlog", linthresh=np.percentile(df["res_m.2"].to_numpy(), 50))

        curr_axs += 1

    if "res_l.2" in df.columns:
        mean_res_l_2 = np.mean(df["res_l.2"].to_numpy())

        p = sns.boxplot(data=df["res_l.2"], ax=axs[curr_axs], orient="v")
        p.set_title(r"$\left( Y - \hat{\ell}_0 \right)^2$" + "\n" + "MEAN = {:.3e}".format(mean_res_l_2))
        p.set_xlabel("")
        p.set_ylabel("")
        p.set_ylim(bottom=0.0, top=np.percentile(df["res_l.2"].to_numpy(), 95))
        p.set_yscale("symlog", linthresh=np.percentile(df["res_l.2"].to_numpy(), 50))

        curr_axs += 1

    if "delta.m.2" in df.columns:
        mean_delta_m_2 = np.mean(df["delta.m.2"].to_numpy())

        p = sns.boxplot(data=df["delta.m.2"], ax=axs[curr_axs], orient="v")
        p.set_title("$(m_0(X) - \hat{m}_0) ^ 2$" + "\n" + "MEAN = {:.3e}".format(mean_delta_m_2))
        p.set_xlabel("")
        p.set_ylabel("")
        p.set_ylim(bottom=0.0, top=np.percentile(df["delta.m.2"].to_numpy(), 95))
        p.set_yscale("symlog", linthresh=np.percentile(df["delta.m.2"].to_numpy(), 50))

        curr_axs += 1

    if "delta.g.2" in df.columns:
        mean_delta_g_2 = np.mean(df["delta.g.2"].to_numpy())

        p = sns.boxplot(data=df["delta.g.2"], ax=axs[curr_axs], orient="v")
        p.set_title("$(g_0(X) - \hat{g}_0) ^ 2$" + "\n" + "MEAN = {:.3e}".format(mean_delta_g_2))
        p.set_xlabel("")
        p.set_ylabel("")
        p.set_ylim(bottom=0.0, top=np.percentile(df["delta.g.2"].to_numpy(), 95))
        p.set_yscale("symlog", linthresh=np.percentile(df["delta.g.2"].to_numpy(), 50))

        curr_axs += 1

    assert curr_axs == nb_columns

    plt.subplots_adjust(wspace=0.25, hspace=0.4)
    plt.savefig(save_as, bbox_inches="tight")
    plt.close()


def plot_model_statistics(history: Dict[str, Any], theta: float, save_as: pathlib.Path):

    required_attr = set(["train loss", "train theta-est", "test loss", "test theta-est"])
    intersection = set(history.keys()).intersection(required_attr)
    if intersection != required_attr:
        raise RuntimeError

    batches = np.arange(start=0, stop=len(history["train-loss"]))
    epochs = np.arange(start=0, stop=len(history["test-loss"]))

    best_test = np.asarray(history["test-loss"]).argmin()
    best_theta_est = history["test-theta-est"][best_test]

    _, axs = plt.subplots(2, 2, figsize=(7.5, 7.5))

    axs[0, 0].plot(batches, history["train-loss"])
    axs[0, 0].set_ylabel("loss")
    axs[0, 0].set_xlabel("batch")
    axs[0, 0].set_title("train loss")
    axs[0, 0].set_yscale("log")

    axs[0, 1].plot(batches, history["train-theta-est"])
    axs[0, 1].set_ylabel("theta estimation")
    axs[0, 1].set_xlabel("batch")
    axs[0, 1].set_title("train theta estimation")
    axs[0, 1].axhline(y=theta, color="red", alpha=0.8, linestyle="--")

    axs[1, 0].plot(epochs, history["test-loss"])
    axs[1, 0].set_ylabel("loss")
    axs[1, 0].set_xlabel("epoch")
    axs[1, 0].set_title("test loss")
    axs[1, 0].set_yscale("log")
    axs[1, 0].axvline(x=best_test, color="red", alpha=0.8, linestyle="--")

    axs[1, 1].plot(batches, history["test-theta-est"])
    axs[1, 1].set_ylabel("test estimation")
    axs[1, 1].set_xlabel("epoch")
    axs[1, 1].set_title("test theta estimation (best= {:.3f})".format(best_theta_est))
    axs[1, 1].axvline(x=best_test, color="red", alpha=0.8, linestyle="--")
    axs[1, 1].axhline(y=theta, color="red", alpha=0.8, linestyle="--")

    plt.suptitle("$\\theta_0$ = " + f"{theta}")

    plt.subplots_adjust(wspace=0.15, hspace=0.50)
    plt.savefig(save_as, bbox_inches="tight")
    plt.close()


def create_axis_grid(num_plots: int, sz: float = 5.0):

    x = math.sqrt(num_plots)
    rows = round(x)
    cols = math.ceil(x)

    fig_height = sz * cols / rows
    fig_width = sz * rows / cols

    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    if rows == 1:
        axs = np.array(axs).reshape((1, cols))

    for r, c in zip(range(rows), range(cols)):
        axs[r, c].set_xticks([])
        axs[r, c].set_yticks([])

    # divider = make_axes_locatable(axs.ravel())
    # cax = divider.append_axes('right', size='5%', pad=0.05)

    return axs, rows, cols


def boxplot_thetas_bias(
    df: pd.DataFrame, n_thetas: int, h_sz: float = 2.5, suptitle: str = "", save_as: Union[str, Path] = None,
):
    w_sz = n_thetas * h_sz

    plt.figure(figsize=(w_sz, h_sz))
    plot = sns.catplot(data=df, col="theta", x="gamma", y="bias", kind="violin", legend_out=True)
    axes = plot.axes.squeeze()
    for ax in axes:
        ax.axhline(y=0.0, color="red", alpha=0.5, linestyle="--")

        # Update title
        t = ax.get_title()
        theta = float(t.split(" = ")[1])
        ax.set_title("$\\theta_0$ = " + f"{theta}")

    # plt.suptitle(suptitle)
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()


def boxplot_cv_gammas(
    df: pd.DataFrame, n_thetas: int, h_sz: float = 2.5, suptitle: str = "", save_as: Union[str, Path] = None,
):

    w_sz = n_thetas * h_sz

    plt.figure(figsize=(w_sz, h_sz))
    plot = sns.catplot(data=df, col="theta", x="gamma", kind="count", legend_out=True)
    axes = plot.axes.squeeze()
    for ax in axes:
        # Update title
        t = ax.get_title()
        theta = float(t.split(" = ")[1])
        ax.set_title("$\\theta_0$ = " + f"{theta}")

    # plt.suptitle(suptitle)
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()


def lineplot_thetas_squared_bias(
    df: pd.DataFrame, y_key: str, y_label: str, suptitle: str = "", save_as: Union[str, Path] = None,
):

    plt.figure()
    x = sns.lineplot(data=df, x="theta", y=y_key, hue="gamma", style="gamma", markers=True, palette="Set1")

    plt.ylabel(y_label)
    plt.xlabel("$\\theta_0$")
    # plt.suptitle(suptitle)
    plt.legend(title="$\gamma$")
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()


def lineplot_thetas_log_squared_y_residual_error(
    df: pd.DataFrame, y_key: str, y_label: str, suptitle: str = "", save_as: Union[str, Path] = None,
):

    plt.figure()
    x = sns.lineplot(
        data=df, x="theta", y=y_key, hue="gamma", style="gamma", hue_norm=mplc.LogNorm(), markers=True, palette="Set1"
    )

    plt.ylabel(y_label)
    plt.xlabel("$\\theta_0$")
    # plt.suptitle(suptitle)
    plt.legend(title="$\gamma$")
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()


def boxplot_final_biases(
    df: pd.DataFrame, n_thetas: int, h_sz: float = 2.5, suptitle: str = "", save_as: Union[str, Path] = None,
):
    df = df.sort_values(by=["theta"])
    df_summary = df.groupby(["theta", "method"]).mean().reset_index()

    w_sz = n_thetas * h_sz

    plt.figure(figsize=(w_sz, h_sz))
    plot = sns.catplot(data=df, col="theta", x="method", y="bias", kind="violin", legend_out=True)
    axes = plot.axes.squeeze()
    for ax in axes:
        # Add horizontal line
        ax.axhline(y=0.0, color="red", alpha=0.5, linestyle="--")

        # Update title with mean values
        t = ax.get_title()
        theta = float(t.split(" = ")[1])
        theta_summary = df_summary[df_summary["theta"] == theta]
        dml_mean_bias = theta_summary[theta_summary["method"] == "DML"]["bias"].item()
        cv_mean_bias = theta_summary[theta_summary["method"] == "SYNC-ML"]["bias"].item()
        summary_str = "mean bias: DML = {:.4f} | SYNC-ML = {:.4f}".format(dml_mean_bias, cv_mean_bias)
        ax.set_title("$\\theta_0$ = " + f"{theta} \n" + summary_str)
        ax.set(xlabel=None)

    # plt.suptitle(suptitle)
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()


import abc
import inspect
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

sns.set_style("darkgrid")
plt.rcParams["font.family"] = "serif"

from argparse import Namespace


class GammaScheduler:
    @classmethod
    def init_from_opts(cls, opts: Namespace):
        return cls(
            epochs=opts.sync_dml_epochs,
            warmup_epochs=opts.sync_dml_warmup_epochs,
            start_gamma=opts.sync_dml_start_gamma,
            end_gamma=opts.sync_dml_end_gamma,
        )

    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 0.0, end_gamma: float = 1.0):
        # self.desc = "{} (epochs={}, warmup_epochs={}, start_gamma={:.3f}, end_gamma={:.3f})".format(
        #     self.__class__.__name__, epochs, warmup_epochs, start_gamma, end_gamma
        # )
        self.desc = "{} ({},{},{:.3f},{:.3f})".format(
            self.__class__.__name__, epochs, warmup_epochs, start_gamma, end_gamma
        )

    @abc.abstractmethod
    def __call__(self, epoch: int) -> float:
        raise NotImplementedError

    def __str__(self):
        return self.desc


class FixedGamma(GammaScheduler):
    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 0.0, end_gamma: float = 1.0):
        super().__init__(epochs, warmup_epochs, start_gamma, end_gamma)

        self.gamma = start_gamma
        self.desc = "{} (gamma={:.2f})".format(self.__class__.__name__, start_gamma)

    def __call__(self, epoch: int) -> float:
        return self.gamma


class LinearGamma(GammaScheduler):
    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 0.0, end_gamma: float = 1.0):
        super().__init__(epochs, warmup_epochs, start_gamma, end_gamma)

        self.gammas = np.linspace(start_gamma, end_gamma, epochs, endpoint=True, dtype=np.float32)
        self.desc = "{} (start_gamma={:.2f}, end_gamma={:.2f})".format(self.__class__.__name__, start_gamma, end_gamma)

    def __call__(self, epoch: int) -> float:
        return self.gammas[epoch]


class GeomGamma(GammaScheduler):
    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 1e-6, end_gamma: float = 1.0):
        super().__init__(epochs, warmup_epochs, start_gamma, end_gamma)

        assert start_gamma > 0.0
        self.gammas = np.geomspace(start_gamma, end_gamma, epochs, endpoint=True, dtype=np.float32)
        self.desc = "{} (start_gamma={:.2f}, end_gamma={:.2f})".format(self.__class__.__name__, start_gamma, end_gamma)

    def __call__(self, epoch: int) -> float:
        return self.gammas[epoch]


class StepGamma(GammaScheduler):
    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 0.0, end_gamma: float = 1.0):

        super().__init__(epochs, warmup_epochs, start_gamma, end_gamma)

        self.start = start_gamma * np.ones(shape=(warmup_epochs,), dtype=np.float32,)
        self.end = end_gamma * np.ones(shape=(epochs - warmup_epochs,), dtype=np.float32,)
        self.gammas = np.concatenate((self.start, self.end))
        self.desc = "{} (warmup_epochs={}, start_gamma={:.2f}, end_gamma={:.2f})".format(
            self.__class__.__name__, warmup_epochs, start_gamma, end_gamma
        )

    def __call__(self, epoch: int) -> float:
        return self.gammas[epoch]


GAMMA_SCHEDULERS = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass) if obj.__module__ is __name__
}


import inspect
import sys
import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.net_m = self.net_m.cuda()
            self.net_l = self.net_l.cuda()

        self.net_m = self.net_m.type(torch.float64)
        self.net_l = self.net_l.type(torch.float64)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        m_x = self.net_m(x).squeeze()
        l_x = self.net_l(x).squeeze()
        return m_x, l_x


class LinearNet(BaseNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net_m = nn.Linear(in_features=in_features, out_features=1)
        self.net_l = nn.Linear(in_features=in_features, out_features=1)

        self.__post_init__()


class NonLinearNet(BaseNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net_m = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 2, out_features=in_features // 4),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 4, out_features=1),
        )

        self.net_l = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 2, out_features=in_features // 4),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 4, out_features=1),
        )

        self.__post_init__()


############################################################

import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    def __init__(self,in_features):
        super(AttentionModule, self).__init__()
        self.channel_attention = nn.Sequential(
        nn.Linear(in_features, int(in_features / 4)),
        nn.ReLU(inplace=True),
        nn.Linear(int(in_features / 4), in_features)
        )

    def forward(self, x):
        attention = self.channel_attention(x)
        return attention

class SoftshareNet(nn.Module):
    def __init__(self, in_features):
        super(SoftshareNet, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
        )
        self.shared_layers=self.shared_layers.cuda()
        self.shared_layers=self.shared_layers.type(torch.float64)
        

        self.attention_m = AttentionModule(in_features).cuda().type(torch.float64)
        self.attention_l = AttentionModule(in_features).cuda().type(torch.float64)
        

        self.net_m = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )
        self.net_m=self.net_m.cuda()

        
        self.net_l = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )
        self.net_l=self.net_l.cuda()
        
        self.net_m=self.net_m.type(torch.float64)
        self.net_l=self.net_l.type(torch.float64)

    def forward(self, x):


        #x=self.attention_l(x)

        shared = self.attention_m(x)
        shared=shared+x
        
        shared = self.shared_layers(x)

        attended_m = self.attention_m(shared)
        attended_m+=shared

        attended_l = self.attention_l(shared)
        attended_l+=shared

        out_m = self.net_m(attended_m).squeeze()
        out_l = self.net_l(attended_l).squeeze()
        
        return out_m, out_l
    
########################################################################################
class SelfAttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttentionModule, self).__init__()


    def forward(self, x):
        _,N=x.shape
        self.query = nn.Linear(N, N).cuda().type(torch.float64)
        self.key = nn.Linear(N, N).cuda().type(torch.float64)
        self.value = nn.Linear(N, N).cuda().type(torch.float64)
        
        #x1=x.permute(1, 0)#(20,250)
        #print(x1.shape)
        Q = self.query(x)#(20,250)
        K = self.key(x)
        V = self.value(x)#(20,250)
        #print(Q.shape)
        #print(x)
        #print(Q)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        #print(attention_scores)
        attention_weights = F.softmax(attention_scores, dim=-1)#(20,20)
        #print(attention_weights.shape)

        attended = torch.matmul(attention_weights, V)#(20,250)
        #attended = attended.permute(1, 0)#(20,250)
        #print(attended.shape)
        return attended

class SoftShareNet(nn.Module):
    def __init__(self, in_features):
        super(SoftShareNet, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
        )
        self.shared_layers=self.shared_layers.cuda()
        self.shared_layers=self.shared_layers.type(torch.float64)
        
        self.attention_m = SelfAttentionModule(in_features).cuda()
        self.attention_l = SelfAttentionModule(in_features).cuda()
        self.attention_l=self.attention_l.type(torch.float64)
        self.attention_m=self.attention_m.type(torch.float64)
        
        self.net_m = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )
        self.net_m=self.net_m.cuda()
        
        self.net_l = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )
        self.net_l=self.net_l.cuda()
        self.net_m=self.net_m.type(torch.float64)
        self.net_l=self.net_l.type(torch.float64)
        

    def forward(self, x):        

        #x=self.attention_l(x)

        shared = self.attention_m(x)
        shared=shared+x
        
        shared = self.shared_layers(x)

        attended_m = self.attention_m(shared)
        attended_m+=shared

        attended_l = self.attention_l(shared)
        attended_l+=shared

        out_m = self.net_m(attended_m).squeeze()
        out_l = self.net_l(attended_l).squeeze()
        
        return out_m, out_l
    
############################################################################
    
class ISelfAttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(ISelfAttentionModule, self).__init__()


    def forward(self, x):
        N,_=x.shape
        self.query = nn.Linear(N, N).cuda().type(torch.float64)
        self.key = nn.Linear(N, N).cuda().type(torch.float64)
        self.value = nn.Linear(N, N).cuda().type(torch.float64)
        
        x1=x.permute(1, 0)#(20,250)
        #print(x1.shape)
        Q = self.query(x1)#(20,250)
        K = self.key(x1)
        V = self.value(x1)#(20,250)
        #print(Q.shape)
        #print(x)
        #print(Q)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        #print(attention_scores)
        attention_weights = F.softmax(attention_scores, dim=-1)#(20,20)
        #print(attention_weights.shape)

        attended = torch.matmul(attention_weights, V)#(20,250)
        attended = attended.permute(1, 0)#(20,250)
        #print(attended.shape)
        return attended

class ISoftShareNet(nn.Module):
    def __init__(self, in_features):
        super(ISoftShareNet, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
        )
        self.shared_layers=self.shared_layers.cuda()
        self.shared_layers=self.shared_layers.type(torch.float64)
        
        self.attention_m = ISelfAttentionModule(in_features).cuda()
        self.attention_l = ISelfAttentionModule(in_features).cuda()
        self.attention_l=self.attention_l.type(torch.float64)
        self.attention_m=self.attention_m.type(torch.float64)
        
        self.net_m = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )
        self.net_m=self.net_m.cuda()
        
        self.net_l = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )
        self.net_l=self.net_l.cuda()
        self.net_m=self.net_m.type(torch.float64)
        self.net_l=self.net_l.type(torch.float64)
        

    def forward(self, x):

        #x=self.attention_l(x)

        shared = self.attention_m(x)
        shared=shared+x
        
        shared = self.shared_layers(x)


        attended_m = self.attention_m(shared)
        attended_m+=shared

        attended_l = self.attention_l(shared)
        attended_l+=shared


        out_m = self.net_m(attended_m).squeeze()
        out_l = self.net_l(attended_l).squeeze()
        
        return out_m, out_l
############################################################

class ExpressiveNet(BaseNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net_m = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )

        self.net_l = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )

        self.__post_init__()


############################################################


class SharedNet(nn.Module):
    def __init__(self):
        super().__init__()

    def __post_init__(self):
        if torch.cuda.is_available():
            self.net = self.net.cuda()

        self.net = self.net.type(torch.float64)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        pred = self.net(x).squeeze()
        m_x, l_x = pred[:, 0], pred[:, 1]
        return m_x, l_x


class SharedLinearNet(SharedNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net = nn.Linear(in_features=in_features, out_features=2)

        self.__post_init__()


class SharedNonLinearNet(SharedNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features=in_features // 2, out_features=2),
        )

        self.__post_init__()


Nets = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass) if obj.__module__ is __name__
}
import inspect
import sys
import torch


from torch import Tensor
from typing import Dict


class CorrelationLoss:
    def __init__(self, gamma_scheduler: GammaScheduler, dml_stats: Dict[str, float]):
        self.gamma_scheduler = gamma_scheduler
        self.corr_abs = dml_stats["corr.abs"]
        self.res_m_2 = dml_stats["res_m.2"]
        self.res_l_2 = dml_stats["res_l.2"]

    def __call__(self, d: Tensor, y: Tensor, m_hat: Tensor, l_hat: Tensor, **kwargs):

        ### Residuals
        res_m = d - m_hat
        res_l = y - l_hat

        ### Reconstruction
        res_m_2 = torch.mean(res_m ** 2)
        res_l_2 = torch.mean(res_l ** 2)
        res_corr_abs = torch.absolute(torch.mean(res_m * res_l))

        ### Normalization
        res_m_2 = res_m_2 / self.res_m_2
        res_l_2 = res_l_2 / self.res_l_2
        res_corr_abs = res_corr_abs / self.corr_abs

        ## Loss
        gamma = self.gamma_scheduler(**kwargs)
        loss = res_m_2 + res_l_2 + gamma * res_corr_abs
        return loss


LOSSES = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass) if obj.__module__ is __name__
}

import inspect
import torch.optim as optim


############################################################
OPTIMIZERS_ = optim.__dict__
OPTIMIZERS = {o: OPTIMIZERS_[o] for o in OPTIMIZERS_ if not o.startswith("_") and inspect.isclass(OPTIMIZERS_[o])}
############################################################


############################################################
OPTIMIZERS_PARAMS = {k: {} for k in OPTIMIZERS.keys()}

OPTIMIZERS_PARAMS["SGD"] = dict(momentum=0.9)
OPTIMIZERS_PARAMS["Adam"] = dict(betas=(0.9, 0.999))
############################################################

import argparse
import multiprocessing as mp
import numpy as np
import torch


from pathlib import Path

def add_common_args(parser) -> argparse.ArgumentParser:
    """
    Parse shared args.
    :param parser: argparser object.
    :return: argparser object.
    """
    test = parser.add_argument_group("Test Parameters")
    test.add_argument("--n-processes", type=int, default=(mp.cpu_count() - 1) // 2, help="number of processes to launch")
    test.add_argument("--n-exp", type=int, default=500, help="number of experiments to run")
    test.add_argument("--seed", type=int, default=42, help="random seed")
    test.add_argument("--output-dir", type=Path, default=Path("results"))
    test.add_argument("--name", type=str, default=gen_random_string(5), help="experiment name")
    test.add_argument("--real-theta", type=str, default="0.0", help="true value of theta")

    return parser


def add_data_args(parser) -> argparse.ArgumentParser:
    """
    Parse shared data args.
    :param parser: argparser object.
    :return: argparser object.
    """
    parser.add_argument("--data-type", type=str, default="synthetic", choices=DATA_TYPES.keys())
    parser.add_argument("--nb-features", type=int, default=20, help="number of high-dimensional features")
    parser.add_argument("--nb-observations", type=int, default=2000, help="number of observations")
    parser.add_argument("--sigma-v", type=float, default=1.0, help="V ~ N(0,sigma)")
    parser.add_argument("--sigma-u", type=float, default=1.0, help="U ~ N(0,sigma)")

    syn_data = parser.add_argument_group("Synthetic Data Parameters")
    syn_data.add_argument("--ar-rho", type=float, default=0.8, help="AutoRegressive(rho) coefficient")
    syn_data.add_argument("--majority-s", type=float, default=0.75, help="majority split value")
    syn_data.add_argument("--m0-g0-setup", type=str, default="s1", choices=("s1", "s2", "s3", "s4", "s5"))

    return parser


def add_double_ml_args(parser) -> argparse.ArgumentParser:
    dml = parser.add_argument_group("Double Machine Learning Parameters")
    dml.add_argument("--dml-net", type=str, default="NonLinearNet", choices=Nets.keys())
    dml.add_argument("--dml-lr", type=float, default=0.01, help="learning rate")
    dml.add_argument("--dml-clip-grad-norm", type=float, default=3.0)
    dml.add_argument("--dml-epochs", type=int, default=1200)
    dml.add_argument("--dml-optimizer", type=str, default="SGD", help="torch.optim.Optimizer name")
    return parser


def add_sync_dml_args(parser) -> argparse.ArgumentParser:
    sync_dml = parser.add_argument_group("SYNChronized (Double) Machine Learning Parameters")
    sync_dml.add_argument("--sync-dml-warmup-with-dml", action="store_true", default=False)
    sync_dml.add_argument("--sync-dml-net", type=str, default="NonLinearNet", choices=Nets.keys())
    sync_dml.add_argument("--sync-dml-loss", type=str, default="CorrelationLoss", choices=LOSSES.keys())
    sync_dml.add_argument("--sync-dml-lr", type=float, default=0.01, help="learning rate")
    sync_dml.add_argument("--sync-dml-clip-grad-norm", type=float, default=3.0)
    sync_dml.add_argument("--sync-dml-epochs", type=int, default=1200)
    sync_dml.add_argument("--sync-dml-optimizer", type=str, default="SGD", help="torch.optim.Optimizer name")

    sync_dml.add_argument("--sync-dml-gamma-scheduler", type=str, default="FixedGamma", choices=GAMMA_SCHEDULERS.keys())
    sync_dml.add_argument("--sync-dml-warmup-epochs", type=int, default=500)
    sync_dml.add_argument("--sync-dml-start-gamma", type=float, default=1.0)
    sync_dml.add_argument("--sync-dml-end-gamma", type=float, default=1.0)

    return parser


def set_seed(seed: int):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)


class Parser(object):
    @staticmethod
    def double_ml() -> argparse.Namespace:
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()
        parser = add_common_args(parser)
        parser = add_data_args(parser)
        parser = add_double_ml_args(parser)
        opt = parser.parse_args(args=[])

        set_seed(opt.seed)
        opt.output_dir = opt.output_dir.expanduser()
        opt.output_dir.mkdir(parents=True, exist_ok=True)
        opt.real_theta = float(opt.real_theta)

        return opt

    @staticmethod
    def sync_dml() -> argparse.Namespace:
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()
        parser = add_common_args(parser)
        parser = add_data_args(parser)
        parser = add_sync_dml_args(parser)
        opt = parser.parse_args(args=[])

        set_seed(opt.seed)
        opt.output_dir = opt.output_dir.expanduser()
        opt.output_dir.mkdir(parents=True, exist_ok=True)
        opt.real_theta = float(opt.real_theta)

        return opt

    @staticmethod
    def compare() -> argparse.Namespace:
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()
        parser = add_common_args(parser)
        parser = add_data_args(parser)
        parser = add_double_ml_args(parser)
        parser = add_sync_dml_args(parser)

        regression = parser.add_argument_group("Regression Parameters")
        regression.add_argument(
            "--thetas", type=str, nargs="+", default=[" 0.0", " 10.0"],
        )
        regression.add_argument(
            "--gammas",
            type=str,
            nargs="+",
            default=[
                " 0.000",
                " 0.001",
                " 0.01",
                " 0.1",
                " 1.0",
                " 10.",
                " 100.",
                " 1000.",
            ],
        )
        opt = parser.parse_args(args=[])

        set_seed(opt.seed)
        opt.output_dir = opt.output_dir.expanduser()
        opt.output_dir.mkdir(parents=True, exist_ok=True)
        opt.thetas = [float(theta) for theta in opt.thetas]
        opt.n_thetas = len(opt.thetas)
        opt.gammas = [float(gamma) for gamma in opt.gammas]
        opt.n_gammas = len(opt.gammas)

        return opt


class DoubleMachineLearningPyTorch(BaseModel):

    HISTORY_KEYS = (
        "train loss",
        "train theta hat",
        "train delta_m^2",
        "train delta_l^2",
        "test loss",
        "test theta hat",
        "test delta_m^2",
        "test delta_l^2",
    )

    @staticmethod
    def name():
        return "Double Machine Learning"

    def __init__(self, net_type: str, in_features: int, **kwargs):
        """
        :param net_type: Net type to use.
        :param num_features: Number of features in X.
        """
        super().__init__(**kwargs)

        self.net_type = net_type
        self.in_features = in_features
        self.net = Nets[net_type](in_features=in_features)

    @classmethod
    def init_from_opts(cls, opts: Namespace, **kwargs):
        return cls(net_type=opts.dml_net, in_features=opts.nb_features, **kwargs)

    def fit_params(self, opts: Namespace, **kwargs) -> Dict[str, Any]:
        return dict(
            learning_rate=opts.dml_lr,
            max_epochs=opts.dml_epochs,
            optimizer=opts.dml_optimizer,
            clip_grad_norm=opts.dml_clip_grad_norm,
        )

    def restart(self):
        self.net = Nets[self.net_type](in_features=self.in_features)
        self.reset_history()

    @staticmethod
    def _set_fit_params(**kwargs):
        return dict(
            learning_rate=kwargs.get("learning_rate", 0.001),
            max_epochs=kwargs.get("max_epochs", 1000),
            optimizer=kwargs.get("optimizer", "Adam"),
            clip_grad_norm=kwargs.get("clip_grad_norm", None),
        )

    def fit(self, train: Dict[str, Tensor], test: Dict[str, Tensor], **kwargs):
        """
        Fit the model to the data.
        :param train: a dictionary with train data.
        :param test: a dictionary with test data.
        """
        params = self._set_fit_params(**kwargs)

        loss_fn = torch.nn.MSELoss()
        optimizer_name = params["optimizer"]

        optimizer = OPTIMIZERS[optimizer_name](
            self.net.parameters(), lr=params["learning_rate"], **OPTIMIZERS_PARAMS[optimizer_name]
        )

        custom_temp_dir = "E:/C"
        
        tmpfile = tempfile.NamedTemporaryFile(suffix=".pt", dir=custom_temp_dir, delete=False)
        #tmpfile = tempfile.NamedTemporaryFile(suffix=".pt")
        torch.save(self.net, tmpfile.name)

        test_min_loss = None
        for epoch in range(params["max_epochs"]):
            #print(epoch)
            # Train
            optimizer.zero_grad()
            #print(train["x"].shape)
            m_pred, l_pred = self.net(x=train["x"])
            #print(m_pred.shape)
            #print(m_pred.shape)
            m_loss = loss_fn(train["d"], m_pred)
            l_loss = loss_fn(train["y"], l_pred)
            loss = m_loss + l_loss
            loss.backward()

            if params["clip_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), params["clip_grad_norm"])
            optimizer.step()

            # Train statistics
            theta_hat, _ = est_theta_torch(train["y"].detach(), train["d"].detach(), m_pred.detach(), l_pred.detach())
            self.history["train loss"].append(loss.item())
            #print('loss',loss)
            self.history["train theta hat"].append(theta_hat)
            self.history["train delta_m^2"].append(m_loss.item())
            self.history["train delta_l^2"].append(l_loss.item())
            #print(theta_hat)

            # Evaluation
            with torch.no_grad():
                self.net.eval()
                m_hat, l_hat = self.net(x=test["x"])
                m_loss = loss_fn(test["d"], m_hat).item()
                l_loss = loss_fn(test["y"], l_hat).item()
                test_loss = m_loss + l_loss
                theta_hat, _ = est_theta_torch(
                    test["y"].detach(), test["d"].detach(), m_hat.detach(), l_hat.detach()
                )
                self.net.train()
            #print('m:',m_loss)
            #print('l:',l_loss)
            # Evaluation statistics
            self.history["test loss"].append(test_loss)
            self.history["test theta hat"].append(theta_hat)
            print(theta_hat)
            self.history["test delta_m^2"].append(m_loss)
            self.history["test delta_l^2"].append(l_loss)
            
            #print('m_loss',m_loss)
            #print('l_loss',l_loss)
            #print('test_loss',test_loss)
            #print(test_loss)
            '''
            if test_min_loss is None or test_loss < test_min_loss:
                print('min',test_min_loss)
                test_min_loss = test_loss
                torch.save(self.net, tmpfile.name)
            '''
        torch.save(self.net, tmpfile.name)
        self.net = torch.load(tmpfile.name)
        tmpfile.close()

        return self

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predicts (m(x), l(x)) for a given input x.
        :param x: a tensor of shape (num_samples, num_features).
        :return: m(x) and l(x), each of shape (num_samples, ).
        """
        with torch.no_grad():
            m_hat, l_hat = self.net(x=x)
        return m_hat, l_hat



import tempfile
import torch

from argparse import Namespace

from torch import Tensor
from typing import Any, Dict, Tuple



class SYNChronizedDoubleMachineLearning(BaseModel):

    HISTORY_KEYS = (
        "train loss",
        "train theta hat",
        "train residuals m.2",
        "train residuals l.2",
        "train residuals correlation",
        "test loss",
        "test theta hat",
        "test residuals m.2",
        "test residuals l.2",
        "test residuals correlation",
    )

    @staticmethod
    def name():
        return "Synchronized Double Machine Learning"

    def __init__(self, net_type: str, in_features: int, **kwargs):
        """
        :param net_type: Net type to use.
        :param num_features: Number of features in X.
        """
        super().__init__(**kwargs)

        self.net_type = net_type
        self.in_features = in_features
        self.net = Nets[net_type](in_features=in_features)

    @classmethod
    def init_from_opts(cls, opts: Namespace, **kwargs):
        return cls(net_type=opts.sync_dml_net, in_features=opts.nb_features, **kwargs)

    def fit_params(self, opts: Namespace, dml_stats: Dict[str, float], **kwargs) -> Dict[str, Any]:
        gamma_scheduler = GAMMA_SCHEDULERS[opts.sync_dml_gamma_scheduler].init_from_opts(opts)
        loss_fn = LOSSES[opts.sync_dml_loss](gamma_scheduler=gamma_scheduler, dml_stats=dml_stats)
        return dict(
            loss_fn=loss_fn,
            learning_rate=opts.sync_dml_lr,
            max_epochs=opts.sync_dml_epochs,
            optimizer=opts.sync_dml_optimizer,
            clip_grad_norm=opts.sync_dml_clip_grad_norm,
        )

    def restart(self):
        self.net = Nets[self.net_type](in_features=self.in_features)
        self.reset_history()

    def _set_fit_params(self, **kwargs):
        return dict(
            loss_fn=kwargs.get("loss_fn", None),
            learning_rate=kwargs.get("learning_rate", 0.0001),
            max_epochs=kwargs.get("max_epochs", 1000),
            optimizer=kwargs.get("optimizer", "Adam"),
            clip_grad_norm=kwargs.get("clip_grad_norm", None),
        )

    def fit(self, train: Dict[str, Tensor], test: Dict[str, Tensor], **kwargs):
        """
        Fit the model to the data.
        :param train: a dictionary with train data.
        :param test: a dictionary with test data.
        """
        params = self._set_fit_params(**kwargs)

        loss_fn = params["loss_fn"]
        optimizer_name = params["optimizer"]

        optimizer = OPTIMIZERS[optimizer_name](
            self.net.parameters(), lr=params["learning_rate"], **OPTIMIZERS_PARAMS[optimizer_name]
        )
        

        custom_temp_dir = "E:/C"
        tmpfile = tempfile.NamedTemporaryFile(suffix=".pt", dir=custom_temp_dir, delete=False)
        #tmpfile = tempfile.NamedTemporaryFile(suffix=".pt")
        torch.save(self.net, tmpfile.name)

        test_min_loss = None
        for epoch in range(params["max_epochs"]):
            #print(epoch)
            # Train
            optimizer.zero_grad()
            m_hat, l_hat = self.net(x=train["x"])
            loss = loss_fn(train["d"], train["y"], m_hat, l_hat, epoch=epoch)
            loss.backward()

            if params["clip_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), params["clip_grad_norm"])
            optimizer.step()

            # Train statistics
            theta_hat, _ = est_theta_torch(train["y"].detach(), train["d"].detach(), m_hat.detach(), l_hat.detach())
            self.history["train loss"].append(loss.item())
            self.history["train theta hat"].append(theta_hat)

            res_m = train["d"].detach() - m_hat.detach()
            res_l = train["y"].detach() - l_hat.detach()
            self.history["train residuals m.2"].append(torch.mean(res_m ** 2).item())
            self.history["train residuals l.2"].append(torch.mean(res_l ** 2).item())
            self.history["train residuals correlation"].append(torch.mean(res_m * res_l).item())

            # Evaluation
            with torch.no_grad():
                self.net.eval()
                m_hat, l_hat = self.net(x=test["x"])
                test_loss = loss_fn(test["d"].detach(), test["y"].detach(), m_hat.detach(), l_hat.detach(), epoch=epoch)
                theta_hat, _ = est_theta_torch(test["y"].detach(), test["d"].detach(), m_hat.detach(), l_hat.detach())
                self.net.train()

            self.history["test loss"].append(test_loss.item())
            self.history["test theta hat"].append(theta_hat)

            res_m = test["d"].detach() - m_hat.detach()
            res_l = test["y"].detach() - l_hat.detach()
            self.history["test residuals m.2"].append(torch.mean(res_m ** 2).item())
            self.history["test residuals l.2"].append(torch.mean(res_l ** 2).item())
            self.history["test residuals correlation"].append(torch.mean(res_m * res_l).item())
            
            print(theta_hat)
            
            '''
            if test_min_loss is None or test_loss < test_min_loss:
                print('min',test_min_loss)
                test_min_loss = test_loss
                torch.save(self.net, tmpfile.name)
            '''
        torch.save(self.net, tmpfile.name)
        self.net = torch.load(tmpfile.name)
        tmpfile.close()

        return self

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predicts (m(x), l(x)) for a given input x.
        :param x: a tensor of shape (num_samples, num_features).
        :return: m(x) and l(x), each of shape (num_samples, ).
        """
        with torch.no_grad():
            m_hat, l_hat = self.net(x=x)
        return m_hat, l_hat


import argparse
import numpy as np
import pandas as pd
import torch

from multiprocessing import Pool

from typing import Dict, Tuple
from tqdm import tqdm


def dml_theta_estimator(y: torch.Tensor, d: torch.Tensor, m_hat: torch.Tensor, l_hat: torch.Tensor) -> float:
    import statsmodels.api as sm
    v_hat = d - m_hat
    u_hat = y - l_hat
    v_hat=v_hat.cpu()
    u_hat=u_hat.cpu()
    x_array = v_hat.numpy()
    y_array = u_hat.numpy()
    x_array_with_constant = x_array
    #x_array_with_constant = sm.add_constant(x_array)
    model = sm.OLS(y_array, x_array).fit()
    
    theta_hat = torch.mean(v_hat * u_hat) / torch.mean(v_hat * v_hat)
    print(theta_hat)
    return theta_hat.item(),model


def run_dml(opts, D1, D2, D2_a, D2_b) -> Tuple[Dict, torch.Tensor, Dict]:
    preds = dict(theta=opts.real_theta, gamma=np.nan,  method="DML")

    # Train on D1
    double_ml = DoubleMachineLearningPyTorch.init_from_opts(opts=opts)
    params = double_ml.fit_params(opts=opts)
    double_ml = double_ml.fit(train=D1, test=D2, **params)

    # Predict theta on D2_a
    m_hat, l_hat = double_ml.predict(D2["x"])
    theta_init ,model1= dml_theta_estimator(y=D2["y"], d=D2["d"], m_hat=m_hat, l_hat=l_hat)
    u_hat, v_hat = D2["y"] - l_hat, D2["d"] - m_hat
    
    stats = {
        "corr.abs": torch.mean(torch.absolute(u_hat * v_hat)).item(),
        "res_m.2": torch.mean(v_hat ** 2).item(),
        "res_l.2": torch.mean(u_hat ** 2).item()
    }
    

    # Predict g(X) on D2_b
    g_hat = l_hat - theta_init * m_hat
    dml_theta_for_cv ,model2= dml_theta_estimator(
        y=D2["y"], d=D2["d"], m_hat=m_hat, l_hat=l_hat)

    # Residual error on D2_b
    dml_y_hat = g_hat + dml_theta_for_cv * D2["d"]
    preds["y_res.2"] = torch.mean((D2["y"] - dml_y_hat) ** 2).item()
    
    # Predict final theta on D2
    dml_theta,model3 = dml_theta_estimator(y=D2["y"], d=D2["d"], m_hat=m_hat, l_hat=l_hat)
    preds['model3']=model3.summary()
    preds["U_res.2"] = torch.mean((D2["y"] - l_hat) ** 2).item()
    preds["v_res.2"] = torch.mean((D2["d"] - m_hat) ** 2).item()
    
    
    
    preds["theta_hat"] = dml_theta
    preds["bias"] = dml_theta - opts.real_theta
    

    return preds, g_hat, stats


def run_cdml(opts: argparse.Namespace, D1, D2, D2_a, D2_b, g_hat, dml_stats) -> Dict:
    results = []
    for gamma in opts.gammas:
        opts.sync_dml_start_gamma = gamma

        sync_dml = SYNChronizedDoubleMachineLearning.init_from_opts(opts=opts)
        params = sync_dml.fit_params(opts=opts, dml_stats=dml_stats)

        preds = dict(theta=opts.real_theta, gamma=gamma, method="C-DML")

        # Train on D1
        sync_dml = sync_dml.fit(train=D1, test=D2, **params)

        # Predict theta on D2_a
        m_hat, l_hat = sync_dml.predict(D2["x"])
        sync_dml_theta_for_cv,model1 = dml_theta_estimator(y=D2["y"], d=D2["d"], m_hat=m_hat, l_hat=l_hat)

        # Residual error on D2_b
        g_hat = l_hat - sync_dml_theta_for_cv * m_hat
        sync_dml_y_hat= g_hat + sync_dml_theta_for_cv * D2["d"]
        preds["y_res.2"] = torch.mean((D2["y"] - sync_dml_y_hat) ** 2).item()

        # Predict final theta on D2
        sync_dml_theta,model2 = dml_theta_estimator(y=D2["y"], d=D2["d"], m_hat=m_hat, l_hat=l_hat)
        preds["U_res.2"] = torch.mean((D2["y"] - l_hat) ** 2).item()
        print("gamma",gamma)
        print("y_res.2",torch.mean((D2["y"] - sync_dml_y_hat) ** 2).item())
        print("U_res.2",torch.mean((D2["y"] - l_hat) ** 2).item())
        print("v_res.2",torch.mean((D2["d"] - m_hat) ** 2).item())
        preds["v_res.2"] = torch.mean((D2["d"] - m_hat) ** 2).item()
        preds["Y_res.2"] = torch.sum((D2["y"] - l_hat) ** 2)
        preds["gamma"]=gamma
        print(model2.summary())
        preds["model"]=model2.summary()
        preds["theta_hat"] = sync_dml_theta
        preds["bias"] = sync_dml_theta - opts.real_theta

        results.append(preds)

    results = pd.DataFrame(results)
    results = results.sort_values(by="y_res.2", ascending=True)
    preds = results.iloc[0].squeeze().to_dict()

    return preds

class DataSimulator:
    def __init__(self, n, k, l, n_good, n_bad, n_mediatory, n_med,effect_X_to_D=0.8, effect_X_to_Y=0.2):
        self.n = n
        self.k = k
        self.l = l
        self.n_good = n_good
        self.n_bad = n_bad
        self.n_mediatory = n_mediatory
        self.n_med=n_med
        self.effect_X_to_D = effect_X_to_D
        self.effect_X_to_Y = effect_X_to_Y

    def simulate_data(self):
        X = np.random.randn(self.n, self.k, self.l)
        D = np.random.randn(self.n, self.l)
        Y = np.random.randn(self.n, self.l)
        
        if self.n_bad==0 and self.n_mediatory ==0 and self.n_med==0:
            for t in range(self.l):
                for i in range(self.n_good):
                    D[:,t]+=X[:, i , t]*self.effect_X_to_D
                    Y[:,t]+=X[:, i , t]*self.effect_X_to_Y 
                Y[:, t] +=D[:, t]*t
                
        if self.n_bad > 0:
            for t in range(self.l):
                for i in range(self.n_bad):
                    X[:, self.n_good+i , t] += D[:, t]*self.effect_X_to_D + Y[:, t]*self.effect_X_to_Y
                for i in range(self.n_good):
                    D[:,t]+=X[:, i , t]*self.effect_X_to_D
                    Y[:,t]+=X[:, i , t]*self.effect_X_to_Y 
                Y[:, t] +=D[:, t]*t
                
        if self.n_mediatory > 0:
            for t in range(self.l):
                for i in range(self.n_good):
                    D[:,t]+=X[:, i , t]*self.effect_X_to_D
                    Y[:,t]+=X[:, i , t]*self.effect_X_to_Y             
                for i in range(self.n_mediatory):
                    Y[:,t]+=X[:,self.n_good+i, t]*self.effect_X_to_Y 
                    X[:,self.n_good+i,t]+= D[:,t]*self.effect_X_to_D
                Y[:, t] +=D[:, t]*t
                
        if self.n_med > 0:
            for t in range(self.l):
                for i in range(self.n_good):
                    D[:,t]+=X[:, i , t]*self.effect_X_to_D
                    Y[:,t]+=X[:, i , t]*self.effect_X_to_Y             
                for i in range(self.n_med):
                    Y[:,t]+=X[:,self.n_good+i, t]*self.effect_X_to_Y 
                    #D[:,t]+= X[:,self.n_good+i,t]*self.effect_X_to_D

                Y[:, t] +=D[:, t]*t
        return X, D, Y

simulator = DataSimulator(n=1000, k=20, l=11, n_good=15, n_bad=5, n_mediatory=0, n_med=0)
X1, D1, Y1 = simulator.simulate_data()

X=X1[:,0:20,1].squeeze()
D=D1[:,1].squeeze()
Y=Y1[:,1].squeeze()


def run_experiment(opts: argparse.Namespace) -> Dict[str, Dict]:
    import pandas as pd
    import torch
    import numpy as np
    from sklearn.model_selection import train_test_split
    #data = pd.read_excel("data.xlsx")
    #df=data


    #Y = df_2006_2011['net_tfa'] +  np.random.randn(128)

    #D = df_2006_2011['e401']

    X_numpy = X
    #X_numpy = X.values
    Y_numpy = Y.values
    D_numpy = D.values
    
    X_numpy = X_numpy.astype(np.float64)
    Y_numpy = Y_numpy.astype(np.float64)
    D_numpy = D_numpy.astype(np.float64)

    X_tensor = torch.from_numpy(X_numpy).to('cuda:0')
    Y_tensor = torch.from_numpy(Y_numpy).to('cuda:0')
    D_tensor = torch.from_numpy(D_numpy).to('cuda:0')
    
    X_tensor = torch.tensor(X_tensor)
    D_tensor = torch.tensor(D_tensor)
    Y_tensor = torch.tensor(Y_tensor)
    (
            x_train,
            x_test,
            d_train,
            d_test,
            y_train,
            y_test,
            ) = train_test_split(X_tensor, D_tensor,Y_tensor, train_size=0.5, random_state=42, shuffle=True)

    D1 = {"x": x_train, "d": d_train, "y": y_train}
    D2 = {"x": x_test, "d": d_test, "y": y_test}
    (
            x_train1,
            x_test1,
            d_train1,
            d_test1,
            y_train1,
            y_test1,
            ) = train_test_split(D2["x"],D2["d"], D2["y"], train_size=0.5, random_state=42, shuffle=True)
    
    D2_a = {"x": x_train1, "d": d_train1, "y": y_train1}
    D2_b = {"x": x_test1, "d": d_test1, "y": y_test1}
    dml_results, g_hat, dml_stats = run_dml(opts, D1, D2, D2_a, D2_b)
    cdml_results = run_cdml(opts, D1, D2, D2_a, D2_b, g_hat, dml_stats)

    return {
        "dml_results": dml_results,
        "cdml_results": cdml_results,
    }

def run_cv(opts: argparse.Namespace) -> pd.DataFrame:

    pbar = tqdm(total=1, desc=f"running C-DML")

    def _update(*a):
        pbar.update()
    
    results=run_experiment(opts)
    print(results)

    results = [result["dml_results"] for result in results] + [result["cdml_results"] for result in results]
    return pd.DataFrame(results)



import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import multiprocessing




from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")


plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)
x = sns.color_palette("Set1")
palette = {"DML": x[0], "C-DML": x[1]}


if __name__ == '__main__':
    HTML("""<style>
    .output_png {
        display: table-cell;
        text-align: center;
        vertical-align: middle;
    }
    </style>""")

    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    x = sns.color_palette("Set1")
    palette = {"DML": x[0], "C-DML": x[1]}

    #
    opts = Parser.compare()
    opts.data_type = 'synthetic'
    opts.majority_s = 0.8
    #ExpressiveNet，NonLinearNet，SharedNonLinearNet,SoftshareNet,SoftShareNet,ISoftShareNet
    opts.dml_net = 'NonLinearNet'
    opts.sync_dml_net = 'SharedNonLinearNet'
    #     SharedNonLinearNet      SoftshareNet
    opts.dml_lr=0.05
    opts.sync_dml_lr = 0.05
    opts.n_exp = 3
    opts.nb_features=20
################################
##########################################################################################################################
    

    rhos = (0.1, 0.9)
    opts.real_theta = 1
    opts.gammas = [0,0.1,0.5,1.0,1.5,2]
    #opts.gammas = [0.0, 0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4 ,1.5,1.6,1.7,1.8,1.9,2,5]
    opts.n_gammas = len(opts.gammas)

    results0_df = pd.DataFrame()
    for rho in rhos:

        opts_copy = copy.deepcopy(opts)
        opts_copy.ar_rho = rho

        results0_df_rho = run_cv(opts_copy)
        results0_df_rho["rho"] = rho
        results0_df = pd.concat([results0_df, results0_df_rho], ignore_index=True,)
    rhos = (0.1,0.9)
    opts.real_theta = 100.0
    opts.gammas = [0.0, 0.1, 0.5, 1.0, 1.5]
    #opts.gammas = [0.0, 0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9, 1.0,1.1,1.2,1.3,1.4 ,1.5,1.6,1.7,1.8,1.9,2,5]
    opts.n_gammas = len(opts.gammas)

    results1_df = pd.DataFrame()
    for rho in rhos:

        opts_copy = copy.deepcopy(opts)
        opts_copy.ar_rho = rho

        results1_df_rho = run_cv(opts_copy)
        results1_df_rho["rho"] = rho
        results1_df = pd.concat([results1_df, results1_df_rho], ignore_index=True,)
        hue_order = ["DML", "C-DML"]
    data = [(0.0, results0_df), (1.0, results1_df)]

    with sns.axes_style("whitegrid"):
        _, axs = plt.subplots(len(data), 1, figsize=(9, 4 * len(data)))

        for ax, (theta, df) in zip(axs, data):

            boxplots = sns.boxplot(
                data=df,
                x="rho",
                y="bias",
                hue="method",
                hue_order=hue_order,
                palette=palette,
                ax=ax,
                linewidth=1.5,
                boxprops=dict(facecolor="none"),
            )

            for box, key in zip(boxplots.artists, len(rhos) * hue_order):
                box.set_edgecolor(palette[key])

            ax.set(facecolor="white")
            ax.axhline(y=0.0, linestyle="-", linewidth=0.5, color="black")
            ax.set_title(r"$\theta=$" + str(int(theta)), fontsize=20)
            ax.set_xticklabels([r"$\rho=$" + str(rho) for rho in rhos], fontsize=20)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.legend_.set_title(None)
            ax.grid(visible=False)

            ax_twin = ax.twinx()
            ax_twin.set_ylabel(r"$\Delta \hat{\theta}$", fontsize=20, rotation=270, labelpad=25)
            ax_twin.set_yticks([])
            ax_twin.set_yticklabels([])
            ax_twin.grid(visible=False)

        axs[0].get_legend().remove()

        for ax in axs:
            ax.spines["top"].set(visible=True, color="black")
            ax.spines["bottom"].set(visible=True, color="black")
            ax.spines["left"].set(visible=True, color="black")
            ax.spines["right"].set(visible=True, color="black")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show();
    
