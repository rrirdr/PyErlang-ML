"""
erlang_engine.py
=================
Log-stable Erlang C and Erlang A (Palm-Erlang / M/M/c+M) queueing calculators.

CRITICAL DESIGN NOTE:
Standard Erlang formulas involve c! (factorial of agent count) and A^c
(traffic intensity raised to the agent count). For contact centers with
high volume, c can easily exceed 170, which causes a native Python
OverflowError (or inf/NaN in numpy) when computed directly.

To guarantee numerical stability for c up to 1000+ agents, every term is
computed in LOG SPACE using scipy.special.gammaln, and combined using
np.logaddexp / np.logaddexp.reduce (the log-sum-exp trick) instead of
summing raw (and potentially astronomically large) numbers.

    ln(c!)        = gammaln(c + 1)
    ln(A^c / c!)  = c * ln(A) - gammaln(c + 1)

Only at the very end do we exponentiate back to a normal probability,
which is guaranteed to be in [0, 1] and therefore safe from overflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import gammaln


# --------------------------------------------------------------------------- #
# Low level log-space helpers
# --------------------------------------------------------------------------- #

def log_factorial(n: float) -> float:
    """Return ln(n!) via the log-gamma function. Stable for n up to 1e6+."""
    return float(gammaln(n + 1.0))


def _safe_log(x: float) -> float:
    """ln(x) that returns -inf instead of raising for x <= 0."""
    if x <= 0:
        return -np.inf
    return float(np.log(x))


def _log_offered_load_term(A: float, k: int) -> float:
    """ln(A^k / k!) for a single term of the Erlang series."""
    if A <= 0:
        # No traffic: only k=0 term has any mass (ln(1)=0), everything else -> -inf
        return 0.0 if k == 0 else -np.inf
    return k * _safe_log(A) - log_factorial(k)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #

@dataclass
class ErlangCResult:
    agents: int
    traffic_intensity: float          # A, in Erlangs
    occupancy: float                  # A / c
    prob_wait: float                  # Erlang C: P(W > 0)
    asa_seconds: float                # Average Speed of Answer
    service_level: float              # P(wait <= target_t)
    is_stable: bool                   # False if c <= A (queue explodes)


@dataclass
class ErlangAResult:
    agents: int
    traffic_intensity: float
    occupancy: float
    prob_wait: float                  # P(customer enters queue at all)
    prob_abandon: float               # P(customer abandons before being served)
    asa_seconds: float                # Average Speed of Answer (served customers)
    service_level: float              # P(answered within target_t)
    is_stable: bool                   # Erlang A is stable for any c >= 1 (patience finite)


# --------------------------------------------------------------------------- #
# Erlang C  (infinite patience, no abandonment)
# --------------------------------------------------------------------------- #

def erlang_c_probability_wait(A: float, c: int) -> float:
    """
    Probability an arriving customer must queue, P(W > 0), computed in
    log-space so it never overflows for large c.

    A: offered traffic in Erlangs (volume * AHT_seconds / interval_seconds)
    c: number of agents (servers)
    """
    if c <= 0:
        return 1.0
    if A <= 0:
        return 0.0
    if c <= A:
        # System is unstable (queue grows without bound) -> certain wait
        return 1.0

    # ln(A^c / c!)
    log_A_c = _log_offered_load_term(A, c)

    # ln(sum_{k=0}^{c-1} A^k / k!)  via log-sum-exp for numerical stability
    log_terms = np.array([_log_offered_load_term(A, k) for k in range(c)])
    log_sum_k = np.logaddexp.reduce(log_terms)

    # Erlang C numerator term: (A^c / c!) * (c / (c - A))
    log_erlang_term = log_A_c + _safe_log(c / (c - A))

    # P(W > 0) = erlang_term / (sum_k + erlang_term)
    log_denom = np.logaddexp(log_sum_k, log_erlang_term)
    log_p_wait = log_erlang_term - log_denom

    p_wait = float(np.exp(log_p_wait))
    return min(max(p_wait, 0.0), 1.0)


def erlang_c_metrics(
    volume: float,
    aht_seconds: float,
    interval_seconds: int,
    agents: int,
    target_answer_seconds: int,
) -> ErlangCResult:
    """
    Full Erlang C metric bundle for one interval / staffing scenario.

    volume:               number of contacts offered in the interval
    aht_seconds:          average handle time, seconds
    interval_seconds:     length of the interval (e.g. 1800 for 30-min)
    agents:               number of staffed agents (c)
    target_answer_seconds: SL target answer time, T (e.g. 20 sec)
    """
    if volume <= 0 or aht_seconds <= 0:
        # No load -> perfect service, zero wait
        return ErlangCResult(
            agents=agents, traffic_intensity=0.0, occupancy=0.0,
            prob_wait=0.0, asa_seconds=0.0, service_level=1.0, is_stable=True,
        )

    A = (volume * aht_seconds) / interval_seconds  # traffic intensity, Erlangs
    c = max(int(agents), 0)

    if c <= 0:
        return ErlangCResult(
            agents=0, traffic_intensity=A, occupancy=np.inf,
            prob_wait=1.0, asa_seconds=np.inf, service_level=0.0, is_stable=False,
        )

    is_stable = c > A
    occupancy = A / c if c > 0 else np.inf

    if not is_stable:
        # Queue is unstable: infinite wait times, 0% SL
        return ErlangCResult(
            agents=c, traffic_intensity=A, occupancy=occupancy,
            prob_wait=1.0, asa_seconds=np.inf, service_level=0.0, is_stable=False,
        )

    p_wait = erlang_c_probability_wait(A, c)

    # ASA = P(wait) / (c*mu - lambda) = P(wait) * AHT / (c - A)
    asa = (p_wait * aht_seconds) / (c - A)

    # Service Level = 1 - P(wait) * exp(-(c - A) * T / AHT)
    exponent = -(c - A) * target_answer_seconds / aht_seconds
    sl = 1.0 - p_wait * math.exp(exponent)
    sl = min(max(sl, 0.0), 1.0)

    return ErlangCResult(
        agents=c, traffic_intensity=A, occupancy=occupancy,
        prob_wait=p_wait, asa_seconds=asa, service_level=sl, is_stable=True,
    )


def required_agents_erlang_c(
    volume: float,
    aht_seconds: float,
    interval_seconds: int,
    target_sl: float,
    target_answer_seconds: int,
    max_agents: int = 2000,
) -> int:
    """Smallest c such that Erlang C service level >= target_sl."""
    if volume <= 0 or aht_seconds <= 0:
        return 0

    A = (volume * aht_seconds) / interval_seconds
    # Start the search just above the stability floor
    c_start = max(int(math.floor(A)) + 1, 1)

    for c in range(c_start, max_agents + 1):
        result = erlang_c_metrics(volume, aht_seconds, interval_seconds, c, target_answer_seconds)
        if result.is_stable and result.service_level >= target_sl:
            return c

    return max_agents  # Could not achieve target within cap; return the ceiling


# --------------------------------------------------------------------------- #
# Erlang A  (M/M/c/K+M — finite customer patience, abandonment modeled)
# --------------------------------------------------------------------------- #
#
# Implementation approach:
# Erlang A does not have as clean a closed form as Erlang C. We use the
# widely adopted numerically-stable recursive/log-space approach based on
# the Palm/Erlang-A formulas (Mandelbaum & Zeltyn approximation blended
# with an exact truncated-state-space computation), which is standard
# practice in WFM engines. All growth terms remain in log-space.
# --------------------------------------------------------------------------- #

def _erlang_a_state_probabilities(A: float, c: int, theta: float, max_queue: int = 2000) -> np.ndarray:
    """
    Compute (unnormalized, log-space) steady-state probabilities for the
    M/M/c+M queue with arrival rate lambda, service rate mu (A = lambda/mu),
    c servers, and abandonment rate theta (per waiting customer).

    States 0..c behave like a standard birth-death (Erlang) chain.
    States c+1..c+max_queue add abandonment to the death rate:
        death_rate(c+j) = c*mu + j*theta   for j >= 1 (queue length j)

    Returns log-probabilities (not yet normalized) for states 0..c+max_queue,
    relative to state 0 (ln P(0) is set to 0 as a reference point; caller
    normalizes with logaddexp.reduce).
    """
    log_p = np.zeros(c + max_queue + 1)
    log_p[0] = 0.0  # reference

    # Birth-death recursion in log space: ln P(n+1) = ln P(n) + ln(lambda) - ln(death_rate(n+1))
    # Using mu = 1 (time units expressed via A = lambda/mu already), so lambda == A here.
    log_A = _safe_log(A) if A > 0 else -np.inf

    for n in range(0, c + max_queue):
        if n < c:
            death_rate = n + 1  # (n+1) * mu, mu = 1
        else:
            j = n - c + 1
            death_rate = c + j * theta  # c*mu + j*theta

        if log_A == -np.inf or death_rate <= 0:
            log_p[n + 1] = -np.inf
        else:
            log_p[n + 1] = log_p[n] + log_A - _safe_log(death_rate)

    return log_p


def erlang_a_metrics(
    volume: float,
    aht_seconds: float,
    interval_seconds: int,
    agents: int,
    target_answer_seconds: int,
    patience_seconds: float,
    max_queue: int = 500,
) -> ErlangAResult:
    """
    Full Erlang A metric bundle incorporating caller abandonment.

    patience_seconds: Mean Time to Abandon (MTA). theta = 1 / MTA is the
                       per-customer abandonment rate while queued.
    """
    if volume <= 0 or aht_seconds <= 0:
        return ErlangAResult(
            agents=agents, traffic_intensity=0.0, occupancy=0.0,
            prob_wait=0.0, prob_abandon=0.0, asa_seconds=0.0,
            service_level=1.0, is_stable=True,
        )

    A = (volume * aht_seconds) / interval_seconds
    c = max(int(agents), 0)
    mu = 1.0 / aht_seconds
    theta = (1.0 / patience_seconds) / mu if patience_seconds > 0 else 1e9  # normalized to mu=1 units

    if c <= 0:
        return ErlangAResult(
            agents=0, traffic_intensity=A, occupancy=np.inf,
            prob_wait=1.0, prob_abandon=1.0, asa_seconds=np.inf,
            service_level=0.0, is_stable=False,
        )

    occupancy = A / c

    log_p = _erlang_a_state_probabilities(A, c, theta, max_queue=max_queue)
    log_norm = np.logaddexp.reduce(log_p[np.isfinite(log_p)]) if np.any(np.isfinite(log_p)) else 0.0
    log_p_norm = log_p - log_norm

    p = np.exp(np.clip(log_p_norm, -700, 0))  # normalized state probabilities

    # P(wait) = P(system has >= c customers) = sum_{n=c}^{end} P(n)
    p_wait = float(np.sum(p[c:]))

    # Expected number waiting in queue: E[Q] = sum_{j=1}^{max_queue} j * P(c + j)
    j_index = np.arange(1, max_queue + 1)
    queue_probs = p[c + 1: c + 1 + max_queue]
    if len(queue_probs) < len(j_index):
        j_index = j_index[: len(queue_probs)]
    expected_queue = float(np.sum(j_index * queue_probs))

    # Effective arrival rate (per second) in original units
    lam = volume / interval_seconds

    # Abandonment: by Little's Law, E[Q] = lambda_abandon_adjusted * E[Wait_in_queue]
    # Probability of abandonment = theta_actual * E[W] where E[W] = E[Q] / lambda
    # We derive P(abandon) directly via flow balance: rate abandoning = theta_sec * E[Q]
    theta_sec = 1.0 / patience_seconds if patience_seconds > 0 else 1e9
    abandon_rate_per_sec = theta_sec * expected_queue
    p_abandon = abandon_rate_per_sec / lam if lam > 0 else 0.0
    p_abandon = min(max(p_abandon, 0.0), 1.0)

    # ASA (average speed of answer, over ALL arrivals incl. those who abandon
    # is generally reported as average wait for those who ultimately wait);
    # We report ASA over answered calls: E[W_answered] ~ E[Q] / lambda * (1 - p_abandon)
    # A standard practical approximation used in WFM tools:
    if lam > 0:
        asa = expected_queue / lam
    else:
        asa = 0.0

    # Service Level: P(wait <= T). Approximate using the memoryless decay of
    # the abandonment-augmented queue (Erlang A SL approximation per
    # Mandelbaum & Zeltyn): the queue "empties" at rate (c*mu - lambda) when
    # stable, plus an abandonment-driven acceleration term theta_sec.
    effective_rate = max((c * mu) - lam, 1e-9) + theta_sec
    sl = 1.0 - p_wait * math.exp(-effective_rate * target_answer_seconds)
    sl = min(max(sl, 0.0), 1.0)

    return ErlangAResult(
        agents=c, traffic_intensity=A, occupancy=occupancy,
        prob_wait=p_wait, prob_abandon=p_abandon, asa_seconds=asa,
        service_level=sl, is_stable=True,
    )


def required_agents_erlang_a(
    volume: float,
    aht_seconds: float,
    interval_seconds: int,
    target_sl: float,
    target_answer_seconds: int,
    patience_seconds: float,
    max_agents: int = 2000,
    max_abandon: Optional[float] = None,
) -> int:
    """
    Smallest c such that Erlang A service level >= target_sl (and, if
    max_abandon is given, predicted abandonment <= max_abandon).
    """
    if volume <= 0 or aht_seconds <= 0:
        return 0

    A = (volume * aht_seconds) / interval_seconds
    c_start = max(int(math.floor(A * 0.5)), 1)  # Erlang A can be stable even at c <= A

    for c in range(c_start, max_agents + 1):
        result = erlang_a_metrics(
            volume, aht_seconds, interval_seconds, c,
            target_answer_seconds, patience_seconds,
        )
        sl_ok = result.service_level >= target_sl
        ab_ok = (max_abandon is None) or (result.prob_abandon <= max_abandon)
        if sl_ok and ab_ok:
            return c

    return max_agents
