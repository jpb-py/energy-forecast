# Hierarchical (partial-pooled) model of residual scale by hour-of-day, used to
# turn a fixed point forecast's residuals into hour-varying prediction intervals.
# See notebooks/04_bayesian_intervals.ipynb for prior/posterior diagnostics.
import numpy as np
import pandas as pd
import pymc as pm

from energy_forecast.config import HierarchicalSigmaPriors


def build_hierarchical_sigma_model(
    train_resid: pd.Series,
    train_hour: pd.Series,
    priors: HierarchicalSigmaPriors,
) -> pm.Model:
    coords = {"hour": np.arange(priors.n_hour_buckets)}
    with pm.Model(coords=coords) as model:
        hour_idx = pm.Data("hour_idx", train_hour.to_numpy(), dims="obs_id")
        resid_data = pm.Data("resid_data", train_resid.to_numpy(), dims="obs_id")

        alpha = pm.Normal("alpha", priors.alpha_mu, priors.alpha_sigma)
        tau = pm.HalfNormal("tau", priors.tau_sigma)
        gamma = pm.Normal("gamma", 0.0, tau, dims="hour")
        sigma_h = pm.Deterministic("sigma_h", pm.math.exp(alpha + gamma), dims="hour")

        sigma_obs = sigma_h[hour_idx]
        pm.Normal("resid_obs", mu=0.0, sigma=sigma_obs, observed=resid_data, dims="obs_id")

    return model


def fit_hierarchical_sigma_model(model: pm.Model, priors: HierarchicalSigmaPriors, random_seed: int | None = None):
    with model:
        trace = pm.sample(
            draws=priors.draws,
            tune=priors.tune,
            chains=priors.chains,
            progressbar=False,
            random_seed=random_seed,
        )
    return trace


def predict_interval(
    model: pm.Model,
    trace,
    test_hour: pd.Series,
    priors: HierarchicalSigmaPriors,
    random_seed: int | None = None,
) -> pd.DataFrame:
    test_hour_arr = test_hour.to_numpy()
    with model:
        pm.set_data({
            "hour_idx": test_hour_arr,
            # placeholder observed values, correct length only — sample_posterior_predictive
            # forward-samples resid_obs from its parents and ignores this content
            "resid_data": np.zeros(len(test_hour_arr)),
        })
        ppc = pm.sample_posterior_predictive(
            trace, var_names=["resid_obs"], random_seed=random_seed, progressbar=False
        )

    draws = ppc.posterior_predictive["resid_obs"]
    lower = draws.quantile(priors.interval_low, dim=("chain", "draw")).to_numpy()
    upper = draws.quantile(priors.interval_high, dim=("chain", "draw")).to_numpy()

    return pd.DataFrame(
        {"lower_offset": lower, "upper_offset": upper},
        index=test_hour.index,
    )
