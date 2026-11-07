"""Testes estatísticos de drift (padrão Strategy): PSI, KS, Qui-quadrado e Jensen-Shannon.

Política (documentada): SEVERIDADE = magnitude (PSI/JS); SIGNIFICÂNCIA = KS/Chi² (p-value).
Com n grande o KS acusa diferenças irrelevantes, então uma feature só é "drifted" se a
magnitude passa do limiar E o teste de hipótese é significativo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

EPSILON = 1e-6
N_BINS = 10
MAX_DISCRETE = 20


@dataclass(frozen=True)
class StatResult:
    """Resultado de um teste: estatística, p-value (se houver), decisão e limiar."""

    statistic: float
    p_value: float | None
    drifted: bool
    threshold: float


class Bins:
    """Bins do REFERENCE, calculados UMA vez e reutilizados em todas as janelas (lição 13.8)."""

    def __init__(self, ref: pd.Series | np.ndarray) -> None:
        values = np.asarray(ref, dtype=float)
        uniques = np.unique(values)
        self.discrete = len(uniques) <= MAX_DISCRETE
        if self.discrete:
            self.categories = uniques
            self.edges = np.array([])
        else:
            quantiles = np.quantile(values, np.linspace(0, 1, N_BINS + 1))
            self.edges = np.unique(quantiles)[1:-1]
            self.categories = np.array([])
        self.ref_probs = self.probs(values)

    @property
    def n_bins(self) -> int:
        """Número de bins (inclui o bin 'fora do domínio' nas discretas)."""
        return len(self.categories) + 1 if self.discrete else len(self.edges) + 1

    def probs(self, values: np.ndarray) -> np.ndarray:
        """Proporção por bin; valores fora de [min, max] caem nos bins extremos."""
        values = np.asarray(values, dtype=float)
        if self.discrete:
            idx = np.searchsorted(self.categories, values)
            idx = np.where(np.isin(values, self.categories), idx, len(self.categories))
        else:
            idx = np.digitize(values, self.edges)
        counts = np.bincount(idx, minlength=self.n_bins).astype(float)
        return counts / max(counts.sum(), 1.0)


def psi_from_probs(p: np.ndarray, q: np.ndarray) -> float:
    """PSI = soma((q - p) * ln(q / p)) com epsilon para bins vazios."""
    p, q = np.clip(p, EPSILON, None), np.clip(q, EPSILON, None)
    return float(np.sum((q - p) * np.log(q / p)))


class StatTest(ABC):
    """Interface Strategy: o detector depende só dela."""

    name: str

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    @abstractmethod
    def compute(self, ref: pd.Series, cur: pd.Series, bins: Bins | None = None) -> StatResult:
        """Compara ``cur`` contra ``ref``."""


class PSITest(StatTest):
    """Population Stability Index (< 0,10 estável; 0,10-0,20 moderado; >= 0,20 severo)."""

    name = "psi"

    def compute(self, ref: pd.Series, cur: pd.Series, bins: Bins | None = None) -> StatResult:
        """PSI da janela contra os bins do reference."""
        bins = bins or Bins(ref)
        value = psi_from_probs(bins.ref_probs, bins.probs(np.asarray(cur)))
        return StatResult(value, None, value >= self.threshold, self.threshold)


class KSTest(StatTest):
    """Kolmogorov-Smirnov de duas amostras (drifted se p < alfa)."""

    name = "ks"

    def compute(self, ref: pd.Series, cur: pd.Series, bins: Bins | None = None) -> StatResult:
        """KS de duas amostras; drifted se p < alfa."""
        res = stats.ks_2samp(np.asarray(ref, float), np.asarray(cur, float))
        return StatResult(
            float(res.statistic), float(res.pvalue), res.pvalue < self.threshold, self.threshold
        )


class ChiSquareTest(StatTest):
    """Qui-quadrado de homogeneidade para categóricas (drifted se p < alfa)."""

    name = "chi2"

    def compute(self, ref: pd.Series, cur: pd.Series, bins: Bins | None = None) -> StatResult:
        """Qui-quadrado sobre a tabela de contagens; drifted se p < alfa."""
        table = pd.concat([pd.Series(ref).value_counts(), pd.Series(cur).value_counts()], axis=1)
        table = table.fillna(0).to_numpy()
        table = table[table.sum(axis=1) > 0]
        if table.shape[0] < 2:
            return StatResult(0.0, 1.0, False, self.threshold)
        chi2, p, _, _ = stats.chi2_contingency(table)
        return StatResult(float(chi2), float(p), p < self.threshold, self.threshold)


class JensenShannonTest(StatTest):
    """Distância de Jensen-Shannon (base 2) entre as distribuições binadas, em [0, 1]."""

    name = "js"

    def compute(self, ref: pd.Series, cur: pd.Series, bins: Bins | None = None) -> StatResult:
        """Distância JS (base 2) entre as distribuições binadas."""
        bins = bins or Bins(ref)
        q = bins.probs(np.asarray(cur))
        value = float(jensenshannon(bins.ref_probs + EPSILON, q + EPSILON, base=2))
        return StatResult(value, None, value >= self.threshold, self.threshold)


TESTS: dict[str, type[StatTest]] = {
    "psi": PSITest,
    "ks": KSTest,
    "chi2": ChiSquareTest,
    "js": JensenShannonTest,
}


def create_stattest(name: str, threshold: float) -> StatTest:
    """Factory de testes estatísticos; ValueError para nome inválido."""
    if name not in TESTS:
        raise ValueError(f"teste estatístico inválido: {name!r}. Opções: {sorted(TESTS)}")
    return TESTS[name](threshold)


def bonferroni_alpha(alpha: float, n_tests: int, enabled: bool) -> float:
    """Nível de significância ajustado por comparações múltiplas (opcional)."""
    return alpha / max(n_tests, 1) if enabled else alpha
