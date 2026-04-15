"""
HydroCast — SEIR Physics-Informed Constraint
Differentiable SEIR epidemiological model used as a
physics-informed regulariser during deep learning training.

Forces the neural network predictions to stay within
epidemiologically plausible bounds — key novelty of HydroCast.

Compartments:
    S = Susceptible
    E = Exposed (incubation period)
    I = Infectious
    R = Recovered / Removed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CONFIG, MAHARASHTRA_DISTRICTS

logger = logging.getLogger("hydrocast.seir_constraint")


# ══════════════════════════════════════════════════════════════════
# SEIR MODEL
# ══════════════════════════════════════════════════════════════════

class SEIRModel(nn.Module):
    """
    Differentiable SEIR model for waterborne disease dynamics.

    The model uses learnable transmission (beta), recovery (gamma),
    and incubation (sigma) rates. Water contamination is modelled
    as a time-varying forcing function that modulates beta.

    Parameters
    ----------
    population      : total population of the district
    beta_init       : initial transmission rate (default 0.3)
    gamma_init      : initial recovery rate     (default 0.1)
    sigma_init      : initial incubation rate   (default 0.2)
    dt              : timestep (1/7 year ≈ 1 week)
    """

    def __init__(
        self,
        population: float = 1_000_000,
        beta_init:  float = 0.3,
        gamma_init: float = 0.1,
        sigma_init: float = 0.2,
        dt:         float = 1 / 7,
    ) -> None:
        super().__init__()

        self.N  = population
        self.dt = dt

        # Learnable parameters (constrained to (0, 1))
        self.log_beta  = nn.Parameter(torch.tensor(np.log(beta_init),  dtype=torch.float))
        self.log_gamma = nn.Parameter(torch.tensor(np.log(gamma_init), dtype=torch.float))
        self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma_init), dtype=torch.float))

    @property
    def beta(self)  -> torch.Tensor: return torch.exp(self.log_beta).clamp(0.01, 2.0)
    @property
    def gamma(self) -> torch.Tensor: return torch.exp(self.log_gamma).clamp(0.01, 1.0)
    @property
    def sigma(self) -> torch.Tensor: return torch.exp(self.log_sigma).clamp(0.01, 1.0)

    def estimate_r0(self) -> float:
        """
        Basic reproduction number R0 = beta / gamma.
        R0 > 1 means the outbreak is growing.
        """
        return float((self.beta / self.gamma).detach().cpu())

    def forward(
        self,
        S0: float,
        E0: float,
        I0: float,
        R0: float,
        n_weeks: int,
        water_contamination: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Run SEIR ODE forward for n_weeks using Euler integration.

        Parameters
        ----------
        S0, E0, I0, R0 : initial compartment sizes
        n_weeks        : number of weeks to simulate
        water_contamination : (n_weeks,) tensor of contamination index [0,1]
                              Modulates beta — higher contamination = more transmission

        Returns
        -------
        dict  {S: tensor, E: tensor, I: tensor, R: tensor}
              each shape (n_weeks+1,)
        """
        N      = self.N
        device = self.log_beta.device
        dtype  = self.log_beta.dtype
        S   = torch.tensor(float(S0), dtype=dtype, device=device)
        E   = torch.tensor(float(E0), dtype=dtype, device=device)
        I   = torch.tensor(float(I0), dtype=dtype, device=device)
        R_c = torch.tensor(float(R0), dtype=dtype, device=device)

        S_list, E_list, I_list, R_list = [S], [E], [I], [R_c]

        for t in range(n_weeks):
            # Optionally modulate beta by water contamination level
            beta_t = self.beta
            if water_contamination is not None and t < len(water_contamination):
                contamination = water_contamination[t].to(device=device, dtype=dtype).clamp(0, 1)
                beta_t        = beta_t * (1.0 + contamination)

            # SEIR ODEs (Euler integration)
            new_exposed   = beta_t * S * I / N
            new_infectious= self.sigma * E
            new_recovered = self.gamma * I

            S_new = (S - new_exposed * self.dt).clamp(0, N)
            E_new = (E + (new_exposed - new_infectious) * self.dt).clamp(0)
            I_new = (I + (new_infectious - new_recovered) * self.dt).clamp(0)
            R_new = (R_c + new_recovered * self.dt).clamp(0, N)

            S, E, I, R_c = S_new, E_new, I_new, R_new

            S_list.append(S_new)
            E_list.append(E_new)
            I_list.append(I_new)
            R_list.append(R_new)

        return {
            "S": torch.stack(S_list),
            "E": torch.stack(E_list),
            "I": torch.stack(I_list),
            "R": torch.stack(R_list),
        }

    def compute_seir_loss(
        self,
        predicted_cases: torch.Tensor,
        seir_cases:      torch.Tensor,
        weight:          float = 0.1,
    ) -> torch.Tensor:
        """
        Physics regularisation loss: MSE between DL prediction
        and SEIR-derived expected case trajectory.

        Forces the DL model to stay epidemiologically plausible.

        Parameters
        ----------
        predicted_cases : (batch, horizon) DL model output (normalised)
        seir_cases      : (batch, horizon) SEIR I compartment (normalised)
        weight          : loss scaling factor (default 0.1)

        Returns
        -------
        torch.Tensor  scalar loss
        """
        # Align lengths
        min_len = min(predicted_cases.shape[-1], seir_cases.shape[-1])
        pred    = predicted_cases[..., :min_len]
        seir    = seir_cases[..., :min_len]

        # Normalise SEIR output to [0,1]
        seir_norm = seir / (seir.max() + 1e-8)

        mse_loss = nn.functional.mse_loss(pred, seir_norm)
        return weight * mse_loss

    def fit_to_district(
        self,
        observed_cases: np.ndarray,
        population:     float,
    ) -> dict[str, float]:
        """
        Fit beta, gamma, sigma to historical district case data
        using scipy.optimize.minimize.

        Parameters
        ----------
        observed_cases : (n_weeks,) array of weekly case counts
        population     : district population

        Returns
        -------
        dict  {beta, gamma, sigma, R0, fit_mse}
        """
        from scipy.optimize import minimize

        n_weeks  = len(observed_cases)
        I0       = observed_cases[0]
        S0       = population - I0
        obs_norm = observed_cases / (observed_cases.max() + 1e-8)

        def _objective(params: np.ndarray) -> float:
            beta_p, gamma_p, sigma_p = params
            if any(p <= 0 for p in params):
                return 1e10

            S, E, I, R = S0, I0 * 0.5, I0, 0.0
            sim = []
            for _ in range(n_weeks):
                dE = beta_p * S * I / population
                dI = sigma_p * E
                dR = gamma_p * I
                S  = max(0, S - dE)
                E  = max(0, E + dE - dI)
                I  = max(0, I + dI - dR)
                R  = R + dR
                sim.append(I)

            sim_arr  = np.array(sim)
            sim_norm = sim_arr / (sim_arr.max() + 1e-8)
            return float(np.mean((sim_norm - obs_norm) ** 2))

        result = minimize(
            _objective,
            x0     = [0.3, 0.1, 0.2],
            method = "L-BFGS-B",
            bounds = [(0.01, 2.0), (0.01, 1.0), (0.01, 1.0)],
            options= {"maxiter": 200},
        )

        beta_f, gamma_f, sigma_f = result.x
        r0_f = beta_f / gamma_f

        logger.info(
            f"SEIR fit: beta={beta_f:.3f} gamma={gamma_f:.3f} "
            f"sigma={sigma_f:.3f} R0={r0_f:.2f} MSE={result.fun:.4f}"
        )

        return {
            "beta":    float(beta_f),
            "gamma":   float(gamma_f),
            "sigma":   float(sigma_f),
            "R0":      float(r0_f),
            "fit_mse": float(result.fun),
        }


# ══════════════════════════════════════════════════════════════════
# SEIR REGULARISER — wraps SEIRModel for training integration
# ══════════════════════════════════════════════════════════════════

@dataclass
class DistrictSEIRParams:
    """Fitted SEIR parameters for one district."""
    district: str
    beta:     float
    gamma:    float
    sigma:    float
    R0:       float
    population: float


class SEIRRegularizer:
    """
    Wrapper that manages per-district SEIR models and provides
    regularisation loss for the training loop.

    Usage
    -----
    >>> reg = SEIRRegularizer(district_populations)
    >>> reg.fit_all_districts(df)
    >>> loss_term = reg.get_regularization_loss(predictions, district="Raigad")
    """

    def __init__(
        self,
        district_populations: Optional[dict[str, float]] = None,
        seir_loss_weight:     float = 0.1,
    ) -> None:
        self.loss_weight  = seir_loss_weight
        self.seir_models: dict[str, SEIRModel] = {}
        self.fitted_params: dict[str, DistrictSEIRParams] = {}

        # Default population estimates (Maharashtra 2024 estimates)
        default_pops = {
            "Mumbai City":      3_100_000,  "Mumbai Suburban": 9_400_000,
            "Thane":            11_100_000, "Palghar":         2_990_000,
            "Raigad":           2_640_000,  "Ratnagiri":       1_620_000,
            "Sindhudurg":         870_000,  "Nashik":          6_110_000,
            "Dhule":            2_050_000,  "Nandurbar":       1_650_000,
            "Jalgaon":          4_230_000,  "Ahmednagar":      4_540_000,
            "Pune":            9_429_000,   "Satara":          3_003_000,
            "Sangli":           2_820_000,  "Solapur":         4_320_000,
            "Kolhapur":         3_880_000,  "Aurangabad":      3_701_000,
            "Jalna":            1_960_000,  "Beed":            2_585_000,
            "Osmanabad":        1_657_000,  "Latur":           2_454_000,
            "Nanded":           3_361_000,  "Parbhani":        1_836_000,
            "Hingoli":          1_177_000,  "Buldhana":        2_586_000,
            "Akola":            1_814_000,  "Washim":          1_197_000,
            "Amravati":         2_888_000,  "Yavatmal":        2_776_000,
            "Wardha":           1_300_000,  "Nagpur":          4_653_000,
            "Bhandara":         1_199_000,  "Gondia":          1_323_000,
            "Chandrapur":       2_194_000,  "Gadchiroli":        970_000,
        }
        self.populations = district_populations or default_pops

    def fit_all_districts(
        self,
        df: "pd.DataFrame",
        case_col: str = "cholera_cases",
    ) -> None:
        """
        Fit one SEIR model per district using historical case data.

        Parameters
        ----------
        df       : featured DataFrame with MultiIndex (district, date)
        case_col : case column to fit SEIR on
        """
        import pandas as pd

        if isinstance(df.index, pd.MultiIndex):
            df_flat = df.reset_index()
        else:
            df_flat = df.copy()

        logger.info(f"Fitting SEIR for {len(MAHARASHTRA_DISTRICTS)} districts...")

        for district in MAHARASHTRA_DISTRICTS:
            dist_data = df_flat[df_flat["district"] == district]
            if dist_data.empty or case_col not in dist_data.columns:
                continue

            cases = dist_data[case_col].values.astype(float)
            pop   = self.populations.get(district, 1_000_000)

            seir = SEIRModel(population=pop)
            params = seir.fit_to_district(cases, pop)

            # Update model parameters with fitted values
            with torch.no_grad():
                seir.log_beta.data  = torch.tensor(np.log(params["beta"]), dtype=seir.log_beta.dtype, device=seir.log_beta.device)
                seir.log_gamma.data = torch.tensor(np.log(params["gamma"]), dtype=seir.log_gamma.dtype, device=seir.log_gamma.device)
                seir.log_sigma.data = torch.tensor(np.log(params["sigma"]), dtype=seir.log_sigma.dtype, device=seir.log_sigma.device)

            self.seir_models[district]   = seir
            self.fitted_params[district] = DistrictSEIRParams(
                district   = district,
                beta       = params["beta"],
                gamma      = params["gamma"],
                sigma      = params["sigma"],
                R0         = params["R0"],
                population = pop,
            )

        logger.info(f"SEIR fitted for {len(self.seir_models)} districts.")

    def get_regularization_loss(
        self,
        predictions: torch.Tensor,
        district:    str,
        n_weeks:     int = 4,
    ) -> torch.Tensor:
        """
        Compute SEIR physics regularisation loss for one district.

        Parameters
        ----------
        predictions : (batch, horizon) model output probabilities
        district    : district name
        n_weeks     : forecast horizon

        Returns
        -------
        torch.Tensor  scalar regularisation loss
        """
        if district not in self.seir_models:
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

        seir  = self.seir_models[district]
        pop   = self.populations.get(district, 1_000_000)
        I0    = float(predictions[0, 0].detach() * pop * 0.01)
        S0    = pop - I0

        seir_out = seir.forward(S0=S0, E0=I0 * 0.5, I0=I0, R0=0.0, n_weeks=n_weeks)
        seir_I   = seir_out["I"][1:]   # skip t=0, take forecast weeks

        seir_tensor = seir_I.unsqueeze(0).expand(predictions.shape[0], -1).to(
            device=predictions.device,
            dtype=predictions.dtype,
        )
        return seir.compute_seir_loss(predictions, seir_tensor, weight=self.loss_weight)

    def get_r0_summary(self) -> "pd.DataFrame":
        """Return R0 values for all fitted districts."""
        import pandas as pd
        rows = [
            {"district": d, "R0": p.R0,
             "beta": p.beta, "gamma": p.gamma, "sigma": p.sigma}
            for d, p in self.fitted_params.items()
        ]
        return pd.DataFrame(rows).sort_values("R0", ascending=False)


# ══════════════════════════════════════════════════════════════════
# MAIN — Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test SEIRModel forward pass
    seir = SEIRModel(population=2_640_000, beta_init=0.4, gamma_init=0.1)
    out  = seir.forward(S0=2_639_000, E0=500, I0=500, R0=0, n_weeks=12)

    print("\n── SEIR forward pass ──")
    for k, v in out.items():
        print(f"  {k}: {v[:5].detach().numpy().round(0)}")
    print(f"  R0: {seir.estimate_r0():.2f}")

    # Test physics loss
    preds = torch.rand(4, 4)   # batch=4, horizon=4
    seir_I = out["I"][1:5].unsqueeze(0).expand(4, -1)
    loss  = seir.compute_seir_loss(preds, seir_I, weight=0.1)
    print(f"\n── SEIR physics loss: {loss.item():.6f}")

    # Test fit
    obs   = np.array([10, 18, 32, 55, 88, 120, 98, 70, 45, 28, 15, 8])
    params = seir.fit_to_district(obs, population=2_640_000)
    print(f"\n── Fitted params: {params}")

    # Test SEIRRegularizer
    reg = SEIRRegularizer()
    print(f"\n── Regularizer: {len(reg.populations)} districts with populations")

    print("\n✅ seir_constraint.py smoke test passed.")
