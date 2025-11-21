# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# This script implements the Bayesian-enhanced AoA estimator presented in the corresponding papers:                                   #  
#                                                                                                                                     #
#   θ_i | w, b, τ, x_i  ~  Normal(w^T x_i + b, τ)               (latent AoA)                                                          #
#   μ_phys,i | θ_i      ~  Normal(θ_i, σ_phys,i)                (physics side channel)                                                #
#   y_i | θ_i           ~  Normal(θ_i, σ_y)                     (GT, train-only)                                                      #       
#                                                                                                                                     #
# with global priors:                                                                                                                 #                                                
#    w ~ Normal(0, σ_w I),  b ~ Normal(0, σ_b),  τ ~ HalfNormal(τ_scale).                                                             #
#                                                                                                                                     #
# Both SVI (AutoNormal guide, blocking local θ) and MCMC (NUTS) are provided.                                                         #
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import torch                                                                # PyTorch core.                                           #
import pyro                                                                 # Pyro probabilistic programming.                         #
import numpy                as     np                                       # NumPy for numerical operations.                         #
import pyro.distributions   as     dist                                     # Pyro distributions.                                     #
import pyro.optim           as     pyro_optim                               # Pyro optimization utilities.                            #
from   pyro.infer           import SVI, Trace_ELBO, Predictive, MCMC, NUTS  # Pyro inference utilities.                               # 
from   dataclasses          import dataclass                                # Python dataclass for config.                            # 
from   typing               import Optional, Tuple, Dict, Any               # Typing hints.                                           #
from   pyro.infer.autoguide import AutoNormal                               # AutoNormal guide for SVI.                               #
from   __future__           import annotations                              # Future annotations for Python.                          #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ---------------------------------------------------------- CONFIGURATIONS --------------------------------------------------------- #
Tensor = torch.Tensor # Typing alias for PyTorch tensors.                                                                             #
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------------- MODEL ------------------------------------------------------------- #
@dataclass
class HBLRConfig:
    obs_sigma: float            = 0.1         # σ_y (instrument precision / GT noise)
    w_prior_sd: float           = 1.0         # σ_w
    b_prior_sd: float           = 5.0         # σ_b
    tau_scale: float            = 5.0         # HalfNormal scale for τ
    lr: float                   = 5e-4        # SVI learning rate
    weight_decay: float         = 0.0
    clip_norm: float            = 5.0
    num_steps: int              = 2000        # SVI steps
    batch_size: int             = 256
    num_samples_predictive: int = 1000
    seed: int                   = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    standardize: bool = True        # z-score features
    verbose: bool = True


class HBLR_AoA:
    """
    Physics-informed Hierarchical Bayesian Linear Regression for AoA.

    Model (vectorized over data):
        w ~ N(0, σ_w I)
        b ~ N(0, σ_b)
        τ ~ HalfNormal(τ_scale)

        μ_i = X_i @ w + b
        θ_i ~ N(μ_i, τ)

        (optional) μ_phys,i ~ N(θ_i, σ_phys,i)
        (optional) y_i ~ N(θ_i, σ_y)

    Inference (SVI):
        AutoNormal guide over globals only (block local θ).

    Inference (MCMC):
        NUTS over full joint (including θ).

    Usage:
        model = HBLR_AoA(X_train.shape[1], config=HBLRConfig())
        model.fit_svi(X_train, y_train, mu_phys_train, sigma_phys_train)
        mean, std = model.predict(X_test, mu_phys_test, sigma_phys_test)
    """

    def __init__(self, input_dim: int, config: Optional[HBLRConfig] = None):
        self.config = config or HBLRConfig()
        pyro.set_rng_seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.device = torch.device(self.config.device)
        self.input_dim = int(input_dim)

        # learned / fitted objects
        self._guide = None
        self._svi = None
        self._losses = []
        self._mcmc = None
        self._scaler = None

    def _standardize_X(self, X: Tensor, fit: bool) -> Tensor:
        if not self.config.standardize:
            return X.to(self.device)

        X = X.to(self.device)
        if fit or (self._scaler is None):
            mu = X.mean(dim=0, keepdim=True)
            sd = X.std(dim=0, keepdim=True).clamp_min(1e-8)
            self._scaler = (mu, sd)
        else:
            mu, sd = self._scaler
        return (X - mu) / sd

    def _to_tensor(self, arr, dtype=torch.float32) -> Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype=dtype, device=self.device)
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def _model(self, X: Tensor, mu_phys: Optional[Tensor], sigma_phys: Optional[Tensor], y: Optional[Tensor]) -> None:
        """
        Pyro model. Data plate over N samples.

        Parameters:
            - X          [Tensor] : [N, D] feature matrix
            - mu_phys    [Tensor] : [N] physics AoA estimates (optional)
            - sigma_phys [Tensor] : [N] physics observation noise (optional)
            - y          [Tensor] : [N] GT AoA values (optional)

        Returns:
            - None
        """
        cfg = self.config
        N, D = X.shape

        # Global priors
        w = pyro.sample("w", dist.Normal(
            torch.zeros(D, device=self.device),
            torch.full((D,), cfg.w_prior_sd, device=self.device)
        ).to_event(1))

        b = pyro.sample("b", dist.Normal(
            torch.tensor(0.0, device=self.device),
            torch.tensor(cfg.b_prior_sd, device=self.device)
        ))

        tau = pyro.sample("tau", dist.HalfNormal(torch.tensor(cfg.tau_scale, device=self.device)))

        if w.dim() == 1:
            # Standard case: single weight vector
            mu_lin = (X @ w) + b  # [N]
            tau_expanded = tau
        else:
            # Batched weights
            # Expand X to [1, N, D] and w to [S, 1, D], then do a batched dot product.
            X_exp = X.unsqueeze(0)       # [1, N, D]
            w_exp = w.unsqueeze(1)       # [S, 1, D]
            mu_lin = (X_exp * w_exp).sum(-1)  # [S, N]
            # Add bias
            if b.dim() == 0:
                mu_lin = mu_lin + b
            else:
                mu_lin = mu_lin + b.unsqueeze(-1)
            
            # Expand tau
            if tau.dim() == 0:
                tau_expanded = tau
            else:
                tau_expanded = tau.unsqueeze(-1)

        # Local latent θ with data plate
        with pyro.plate("data", N):
            theta = pyro.sample("theta", dist.Normal(mu_lin, tau_expanded))

            # Physics side channel (optional)
            if mu_phys is not None and sigma_phys is not None:
                pyro.sample("phys_obs", dist.Normal(theta, sigma_phys), obs=mu_phys)

            # Ground-truth observation (optional)
            if y is not None:
                pyro.sample("y_obs", dist.Normal(theta, torch.tensor(cfg.obs_sigma, device=self.device)), obs=y)

    def _make_svi(self, X: Tensor, mu_phys: Optional[Tensor], sigma_phys: Optional[Tensor],
                  y: Optional[Tensor]) -> None:
        """
        Build SVI with AutoNormal guide over GLOBALS only (block local θ).

        Parameters:
            - X          [Tensor] : [N, D] feature matrix
            - mu_phys    [Tensor] : [N] physics AoA estimates (optional)
            - sigma_phys [Tensor] : [N] physics observation noise (optional)
            - y          [Tensor] : [N] GT AoA values (optional)

        Returns:
            - None
        """
        pyro.clear_param_store()

        blocked_model = pyro.poutine.block(self._model, hide=["theta"])

        guide = AutoNormal(blocked_model, init_scale=0.1)
        optim = pyro_optim.ClippedAdam({
            "lr": self.config.lr,
            "weight_decay": self.config.weight_decay,
            "clip_norm": self.config.clip_norm
        })
        svi = SVI(blocked_model, guide, optim, loss=Trace_ELBO())

        self._guide = guide
        self._svi = svi

    def fit_svi(self, X: Tensor, y: Optional[Tensor] = None, mu_phys: Optional[Tensor] = None, sigma_phys: Optional[Tensor] = None) -> Dict[str, Any]:
        """
        Fit with SVI. y is optional (semi-supervised / test-time refinement).

        Parameters:
            - X          [Tensor] : [N, D] feature matrix
            - y          [Tensor] : [N] GT AoA values (optional)
            - mu_phys    [Tensor] : [N] physics AoA estimates (optional)
            - sigma_phys [Tensor] : [N] physics observation noise (optional)

        Returns:
            - results [Dict] : dictionary with training losses and scaler info
        """
        X = self._to_tensor(X)
        X = self._standardize_X(X, fit=True)

        y_t = self._to_tensor(y) if y is not None else None

        mu_t = self._to_tensor(mu_phys) if mu_phys is not None else None
        if mu_t is not None:
            if sigma_phys is None:
                sigma_phys = torch.full_like(mu_t, 2.0)
            sp_t = self._to_tensor(sigma_phys)
        else:
            sp_t = None

        # Build SVI objects
        self._make_svi(X, mu_t, sp_t, y_t)

        # Minibatch loop
        N  = X.shape[0]
        bs = min(self.config.batch_size, N)
        self._losses = []

        for step in range(1, self.config.num_steps + 1):
            start = ((step - 1) * bs) % N
            end   = min(start + bs, N)
            loss  = self._svi.step(X[start:end],
                                  None if mu_t is None else mu_t[start:end],
                                  None if sp_t is None else sp_t[start:end],
                                  None if y_t is None else y_t[start:end])
            loss = loss / max(1, (end - start))
            self._losses.append(float(loss))

            if self.config.verbose and (step == 1 or step % 200 == 0 or step == self.config.num_steps):
                print(f"[SVI] step {step:5d}/{self.config.num_steps}  ELBO: {loss:.4f}")

        return {
            "losses": self._losses,
            "standardize": self.config.standardize,
            "scaler": self._scaler,
        }

    @torch.no_grad()
    def predict(self, X: Tensor, mu_phys: Optional[Tensor] = None, sigma_phys: Optional[Tensor] = None, num_samples: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        Posterior predictive for θ with SVI (guide + model).

        Parameters:
            - X          [Tensor] : [N, D] feature matrix
            - mu_phys    [Tensor] : [N] physics AoA estimates (optional)
            - sigma_phys [Tensor] : [N] physics observation noise (optional)
            - num_samples [int]   : number of θ samples to draw (overrides config)
        Returns:
            - mean [Tensor] : [N] posterior mean of θ
            - std  [Tensor] : [N] posterior std of θ
        """
        if self._guide is None:
            raise RuntimeError("Call fit_svi or fit_mcmc before predict().")

        ns = num_samples or self.config.num_samples_predictive

        X = self._to_tensor(X)
        X = self._standardize_X(X, fit=False)

        mu_t = self._to_tensor(mu_phys) if mu_phys is not None else None
        if mu_t is not None:
            if sigma_phys is None:
                sigma_phys = torch.full_like(mu_t, 2.0)
            sp_t = self._to_tensor(sigma_phys)
        else:
            sp_t = None

        predictive = Predictive(self._model, guide=self._guide, num_samples=ns, return_sites=["theta"])
        out        = predictive(X, mu_t, sp_t, None)
        theta      = out["theta"]  # [S, N]
        mean       = theta.mean(dim=0)
        std        = theta.std(dim=0).clamp_min(1e-9)

        return mean, std

    def fit_mcmc(self, X: Tensor, y: Optional[Tensor] = None, mu_phys: Optional[Tensor] = None, sigma_phys: Optional[Tensor] = None, num_warmup: int = 500,
                 num_samples: int = 1000) -> MCMC:
        """
        Fit with NUTS (asymptotically exact). Stores the MCMC object.

        Parameters:
            - X           [Tensor] : [N, D] feature matrix
            - y           [Tensor] : [N] GT AoA values (optional)
            - mu_phys     [Tensor] : [N] physics AoA estimates (optional)
            - sigma_phys  [Tensor] : [N] physics observation noise (optional)
            - num_warmup  [int]    : number of warmup steps
            - num_samples [int]    : number of posterior samples

        Returns:
            - mcmc [MCMC] : fitted MCMC object
        """
        X = self._to_tensor(X)
        X = self._standardize_X(X, fit=True)

        y_t  = self._to_tensor(y) if y is not None else None
        mu_t = self._to_tensor(mu_phys) if mu_phys is not None else None
        sp_t = self._to_tensor(sigma_phys) if sigma_phys is not None else None

        nuts = NUTS(self._model, target_accept_prob=0.8, max_tree_depth=10)
        mcmc = MCMC(nuts, warmup_steps=num_warmup, num_samples=num_samples, num_chains=1)
        mcmc.run(X, mu_t, sp_t, y_t)
        self._mcmc = mcmc
        return mcmc

    @torch.no_grad()
    def predict_mcmc(self, X: Tensor, mu_phys: Optional[Tensor] = None, sigma_phys: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Predict with MCMC posterior over (w, b, τ). We re-sample θ for new X using posterior samples of (w,b,τ).

        Parameters:
            - X          [Tensor] : [N, D] feature matrix
            - mu_phys    [Tensor] : [N] physics AoA estimates (optional)
            - sigma_phys [Tensor] : [N] physics observation noise (optional)
        Returns:
            - mean [Tensor] : [N] posterior mean of θ
            - std  [Tensor] : [N] posterior std of θ
        """
        if self._mcmc is None:
            raise RuntimeError("Call fit_mcmc() first.")

        X = self._to_tensor(X)
        X = self._standardize_X(X, fit=False)

        mu_t = self._to_tensor(mu_phys) if mu_phys is not None else None
        sp_t = self._to_tensor(sigma_phys) if sigma_phys is not None else None

        # Extract only global parameters from MCMC samples
        mcmc_samples   = self._mcmc.get_samples()
        global_samples = {k: v for k, v in mcmc_samples.items() if k in ["w", "b", "tau"]}
        
        # Form a Predictive with posterior samples from MCMC 
        predictive = Predictive(self._model, posterior_samples=global_samples, return_sites=["theta"])
        out        = predictive(X, mu_t, sp_t, None)
        theta      = out["theta"]

        return theta.mean(dim=0), theta.std(dim=0).clamp_min(1e-9)


def constant_sigma_phys_from_rmse(mu_phys, y, floor: float = 1e-3) -> float:
    """
    Return a single σ_phys as RMSE(mu_phys, y), with a small floor.
    
    Parameters:
        - mu_phys [array-like] : physics-based AoA estimates
        - y       [array-like] : ground-truth AoA values
        - floor   [float]      : minimum σ_phys value to avoid zero

    Returns:
        - sigma_phys [float]   : constant physics observation noise
    """
    mu_phys = np.asarray(mu_phys).ravel()
    y       = np.asarray(y).ravel()
    rmse    = float(np.sqrt(((mu_phys - y) ** 2).mean()))
    return max(rmse, floor)
# =================================================================================================================================== #