# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Contains the necessary functions to train and evaluate a hierarchical Bayesian model for Angle of Arrival (AoA) estimation using    #
# RFID data. The model incorporates encoded physics-informed estimations based on classical antena-array methods and provides         #
# uncertainty quantification for its predictions.                                                                                     #
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                                                 # For file and directory operations.                                        #
import torch                                              # PyTorch for tensor computations and neural networks.                      #
import pyro                                               # Probabilistic programming library.                                        #
import main                    as main                    # Main module for running the analysis and managing experiments.            #
import numpy                   as np                      # Numerical operations.                                                     #
import pyro.distributions      as dist                    # Pyro distributions.                                                       #
import pyro.optim              as pyro_optim              # Pyro optimizers.                                                          #
import networkx                as nx                      # For graph-based analyses (if needed).                                     #
import seaborn                 as sns                     # For enhanced plotting aesthetics.                                         #
import matplotlib              as mpl                     # For plotting settings.                                                    #
import matplotlib.pyplot       as plt                     # For plotting.                                                             #
from   scipy.stats             import kstest              # For Kolmogorov-Smirnov test.                                              #
from   scipy.stats             import norm                # For normal distribution fitting.                                          #
from   sklearn.preprocessing   import StandardScaler      # For feature scaling.                                                      #
from   sklearn.model_selection import train_test_split    # For train-test splitting.                                                 #
from   sklearn.metrics         import mean_absolute_error # MAE metric.                                                               #
from   sklearn.metrics         import mean_squared_error  # RMSE metric.                                                              #
from   pyro.nn                 import PyroModule          # PyroModule base class.                                                    #
from   pyro.nn                 import PyroSample          # PyroSample for defining stochastic layers.                                #
from   pyro.infer              import SVI                 # Stochastic Variational Inference.                                         #
from   pyro.infer              import Trace_ELBO          # ELBO loss function for SVI.                                               #
from   pyro.infer              import Predictive          # Posterior predictive sampling.                                            #
from   pyro.infer.autoguide    import AutoNormal          # Automatic guide for variational inference.                                #
from   cycler                  import cycler              # For custom matplotlib color cycles.                                       # 
from   matplotlib              import rcParams            # For setting matplotlib parameters.                                        #   
mpl.use('Agg')                                            # Use 'Agg' backend for non-interactive plotting (suitable for scripts).    #
import style.style             as     style               # Import custom plotting styles.                                            #     
# =================================================================================================================================== #


# =================================================================================================================================== #
# ----------------------------------------------------- MACHINE LEARNING ANALYSIS --------------------------------------------------- #
SEED = 42
pyro.set_rng_seed(SEED) # For reproducibility - Sets the random number generator for Pyro, ensuring consistent results across runs.   
torch.manual_seed(SEED) # For reproducibility - Sets the random number generator for PyTorch, ensuring consistent results across runs.
np.random.seed(SEED)    # For reproducibility - Sets the random number generator for NumPy, ensuring consistent results across runs.

class BayesianAoARegressor:
    """
    Hierarchical Bayesian AoA regressor.

    Model:
        mean_i = w^T x_i + b
        theta_i ~ Normal(mean_i, tau)
        (optional) mu_phys_i ~ Normal(theta_i, sigma_phys_i)
        y_i ~ Normal(theta_i, obs_sigma)    # single shared scalar noise (small)

    Inference:
        AutoNormal guide over GLOBALS ONLY (hide local 'theta').
    """

    def __init__(self, use_gpu=True, prior_type='ds', feature_mode='full', obs_sigma=0.1):
        """
        Initialize the Bayesian AoA regressor class.

        Parameters:
            - use_gpu      [bool]  : Whether to use GPU, if available.
            - prior_type   [str]   : Type of physics prior to use ('ds', 'weighted', 'music', 'phase', or 'none').
            - feature_mode [str]   : Feature set to use ('full', 'no_distance', 'width_only').
            - obs_sigma    [float] : Observation noise standard deviation (σᵧ).

        Returns:
            - None
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device  = torch.device("cuda" if self.use_gpu else "cpu")
        self.prior_type   = prior_type
        self.feature_mode = feature_mode
        self.obs_sigma    = float(obs_sigma)
        self.model = None
        self.guide = None
        self.scalers = []
        self.feature_names = None
        self.train_summary = None
        print(f"Using device: {self.device}")
        print(f"Prior type: {self.prior_type}")
        print(f"Feature mode: {self.feature_mode}")
        if self.use_gpu:
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------- Feature Extraction -------------------------------------------------- #
    def _extract_features(self, data_manager, include_distance=True, include_width=True):
        """
        Build X, y and physics priors arrays from DataManager. Also returns feature names.

        Parameters:
            - data_manager     [DataManager] : Instance of DataManager with loaded/analyzed data.
            - include_distance [bool]        : Whether to include distance as a feature.
            - include_width    [bool]        : Whether to include tag width as a feature.
        
        Returns:
            - X                [np.ndarray]  : Feature matrix.
            - y                [np.ndarray]  : True angles.
            - feature_names    [list]        : List of feature names.
            - prior_estimates  [dict]        : Dictionary of prior estimates from physics methods.
        """
        # Verify data is analyzed
        if data_manager.results is None:
            data_manager.analyze_all_data(save_results=False)
        # Extract features
        all_features, all_angles = [], []
        prior_estimates = {'ds': [], 'weighted': [], 'music': [], 'phase': []}
        feature_names   = []
        for idx, meta_row in enumerate(data_manager.metadata.iterrows()):
            meta    = meta_row[1]
            signals = data_manager.signal_data[idx]
            phasor1, phasor2 = signals['phasor1'], signals['phasor2']
            rssi1, rssi2     = signals['rssi1'],  signals['rssi2']
            true_angle = data_manager.get_true_angle(meta['D'], meta['W'])
            res_row    = data_manager.results.iloc[idx]
            prior_estimates['ds'].append(res_row['theta_ds'])
            prior_estimates['weighted'].append(res_row['theta_weighted'])
            prior_estimates['music'].append(res_row['theta_music'])
            prior_estimates['phase'].append(res_row['theta_phase'])
            # Core phasor and RSSI features
            phase1_mean = np.angle(np.mean(phasor1))
            phase2_mean = np.angle(np.mean(phasor2))
            phase_diff  = np.angle(np.exp(1j * (phase1_mean - phase2_mean)))
            mag1_mean   = np.mean(np.abs(phasor1))
            mag2_mean   = np.mean(np.abs(phasor2))
            rssi1_mean  = np.mean(rssi1)
            rssi2_mean  = np.mean(rssi2)
            rssi_diff   = rssi1_mean - rssi2_mean
            ph1_r, ph1_i = np.mean(phasor1.real), np.mean(phasor1.imag)
            ph2_r, ph2_i = np.mean(phasor2.real), np.mean(phasor2.imag)
            wavelength = meta['lambda']
            if len(feature_names) == 0:
                feature_names.extend([
                    'phase1_mean','phase2_mean','phase_diff',
                    'mag1_mean','mag2_mean',
                    'rssi1_mean','rssi2_mean','rssi_diff',
                    'phasor1_real_mean','phasor1_imag_mean',
                    'phasor2_real_mean','phasor2_imag_mean',
                    'wavelength'
                ])
            feats = [
                phase1_mean, phase2_mean, phase_diff,
                mag1_mean, mag2_mean,
                rssi1_mean, rssi2_mean, rssi_diff,
                ph1_r, ph1_i, ph2_r, ph2_i,
                wavelength
            ]

            if include_distance:
                feats.append(meta['D'])
                if len(feature_names) == 13:
                    feature_names.append('distance')
            if include_width:
                feats.append(meta['W'])
                if (include_distance and len(feature_names) == 14) or (not include_distance and len(feature_names) == 13):
                    feature_names.append('width')

            all_features.append(feats)
            all_angles.append(true_angle)

        X = np.array(all_features) # Matrix of shape [N, num_features] - Feature matrix.
        y = np.array(all_angles)   # Vector of shape [N]               - True angles.
        for k in prior_estimates:
            prior_estimates[k] = np.array(prior_estimates[k]) # Vector of shape [N] - Prior estimates from physics methods.
    
        self.feature_names = feature_names
        return X, y, feature_names, prior_estimates

    # ---------------------------------------------- Hierarchical Pyro Model ------------------------------------------------ #
    def _build_model(self, input_dim):
        """
        Build the hierarchical Bayesian model using Pyro. The model includes:
            - A linear layer with physics-informed weight priors.
            - A global noise parameter tau (HalfNormal prior).
            - Observation model incorporating optional physics side-channel.
            - Observation noise with fixed small stddev (obs_sigma).

        Parameters:
            - input_dim [int] : Dimensionality of input features.

        Returns:
            - model     [PyroModule] : The constructed Pyro model.
        """
        obs_sigma = self.obs_sigma
        device    = self.device

        class HierarchicalAoA(PyroModule):
            """
            Physics-informed Hierarchical Bayesian AoA model. 

            Model:
                - Linear layer: mean_i = w^T x_i + b
                - theta_i ~ Normal(mean_i, tau)
                - (optional) mu_phys_i ~ Normal(theta_i, sigma_phys_i)
                - y_i ~ Normal(theta_i, obs_sigma)
            Inference:
                - AutoNormal guide over GLOBALS ONLY (hide local 'theta').
            """
            def __init__(self, input_dim):
                super().__init__()
                self.device    = device # Ensure model parameters are on the correct device (CPU/GPU)
                self.obs_sigma = torch.tensor(float(obs_sigma), device=device) # Fixed observation noise stddev

                # Linear layer: Linear regression component of the model - Maps input features to mean AoA
                self.linear = PyroModule[torch.nn.Linear](input_dim, 1).to(device)

                # Weight Prior: Zero mean, physics-informed scale
                w_loc   = torch.zeros((1, input_dim), device=device)
                w_scale = torch.full((1, input_dim), 1.0 / max(1, input_dim**0.5), device=device)
                self.linear.weight = PyroSample(
                    lambda self: dist.Normal(w_loc, w_scale).to_event(1)
                )

                # Bias prior: Scalar Normal
                self.linear.bias = PyroSample(
                    lambda self: dist.Normal(loc=torch.tensor(0.0, device=device),
                                             scale=torch.tensor(5.0, device=device))
                )

            def forward(self, x, mu_phys=None, sigma_phys=None, y=None):
                # STEP 1: Move inputs to correct device if necessary
                if x.device != self.device:
                    x = x.to(self.device)
                # STEP 2: Move optional inputs to correct device if necessary
                if mu_phys is not None and mu_phys.device != self.device:
                    mu_phys = mu_phys.to(self.device)
                if sigma_phys is not None and sigma_phys.device != self.device:
                    sigma_phys = sigma_phys.to(self.device)
                # STEP 3: Move GT to correct device if necessary
                if y is not None and y.device != self.device:
                    y = y.to(self.device)
                # STEP 4: Compute mean predictions
                mean = self.linear(x).squeeze(-1)   # [N]
                # STEP 5: Sample tau directly here (HalfNormal)
                tau_raw = pyro.sample("tau_raw", dist.HalfNormal(scale=torch.tensor(5.0, device=self.device)))
                tau = tau_raw + 1e-3
                # STEP 6: Local latent variable theta for each data point
                with pyro.plate("data", x.shape[0]):
                    # Sample theta_i ~ Normal(mean_i, tau)
                    theta = pyro.sample("theta", dist.Normal(mean, tau))
                    # Optional physics side-channel
                    if mu_phys is not None and sigma_phys is not None:
                        pyro.sample("phys_obs", dist.Normal(theta, sigma_phys), obs=mu_phys)
                    # Observation model
                    if y is not None:
                        pyro.sample("obs", dist.Normal(theta, self.obs_sigma), obs=y)
                    else:
                        pyro.sample("obs", dist.Normal(theta, self.obs_sigma))

                return mean
        # Instantiate model
        model = HierarchicalAoA(input_dim)

        # Ensure params on device
        for _, p in model.named_parameters():
            if p.device != device:
                p.data = p.data.to(device)

        return model

    # -------------------------------------------------------- Training ------------------------------------------------------ #
    def train(self, data_manager, num_epochs=2000, learning_rate=1e-3, batch_size=128, weight_decay=0.0, verbose=True):
        """
        Train the hierarchical Bayesian AoA model.

        Parameters:
            - data_manager [DataManager] : Instance of DataManager with loaded/analyzed data.
            - num_epochs   [int]         : Number of training epochs.
            - learning_rate[int]         : Learning rate for the optimizer.
            - batch_size   [int]         : Mini-batch size for training.
            - weight_decay [float]       : Weight decay (L2 regularization) for the optimizer.
            - verbose      [bool]        : Whether to print training progress.
        
        Returns:
            - train_summary [dict]       : Dictionary containing training results and metrics.
        """
        # Feature Set Up and Extraction
        include_distance = self.feature_mode == 'full'
        include_width    = self.feature_mode in ['full','width_only']
        X, y, feature_names, prior_estimates = self._extract_features(data_manager, include_distance, include_width)

        # Split train/test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Split priors (for comparison/plots)
        prior_splits = {k: train_test_split(prior_estimates[k], test_size=0.1, random_state=42)[1]
                        for k in prior_estimates}

        # Standardize per-feature
        self.scalers = []
        X_train_s = np.zeros_like(X_train)
        X_test_s  = np.zeros_like(X_test)
        for i in range(X_train.shape[1]):
            sc = StandardScaler()
            X_train_s[:, i] = sc.fit_transform(X_train[:, [i]]).ravel()
            X_test_s[:,  i] = sc.transform(X_test[:,  [i]]).ravel()
            self.scalers.append(sc)

        # Torch tensors
        Xtr = torch.tensor(X_train_s, dtype=torch.float32, device=self.device)
        Xte = torch.tensor(X_test_s,  dtype=torch.float32, device=self.device)
        ytr = torch.tensor(y_train,   dtype=torch.float32, device=self.device)
        yte = torch.tensor(y_test,    dtype=torch.float32, device=self.device)

        # Physics side-channel for training/eval (derive from results: use weighted as mu_phys, RMSE as sigma_phys)
        mu_phys_all   = prior_estimates['weighted']
        sigma_phys_all= np.abs(prior_estimates['weighted'] - y)  # crude per-sample sigma;
        # Set sigma_phys to a moderate constant derived from train RMSE of physics:
        physics_rmse = float(np.sqrt(np.mean((prior_estimates['weighted'] - y)**2)))
        sigma_phys_all = np.full_like(y, fill_value=max(physics_rmse, 1e-3))

        # Split physics info. into train/test
        mu_phys_tr, mu_phys_te = train_test_split(mu_phys_all,    test_size=0.1, random_state=42)
        sp_tr,     sp_te       = train_test_split(sigma_phys_all, test_size=0.1, random_state=42)

        # Torch tensors for physics side-channel
        mu_tr = torch.tensor(mu_phys_tr, dtype=torch.float32, device=self.device)
        mu_te = torch.tensor(mu_phys_te, dtype=torch.float32, device=self.device)
        sp_tr = torch.tensor(sp_tr,      dtype=torch.float32, device=self.device)
        sp_te = torch.tensor(sp_te,      dtype=torch.float32, device=self.device)

        # Determine weight prior (mean/std) from chosen physics prior
        if self.prior_type in prior_estimates:
            prior_err = prior_estimates[self.prior_type] - y
            prior_mean = float(np.mean(prior_estimates[self.prior_type]))
            prior_std  = float(max(0.1, np.sqrt(np.mean(prior_err**2))))
            print(f"Using {self.prior_type} prior with weight prior std={prior_std:.4f}")
            self.weight_prior_std = prior_std
        else:
            prior_mean, prior_std = 0.0, 30.0
            print("Using standard broad weight prior N(0,30)")

        pyro.clear_param_store()
        input_dim = Xtr.shape[1]
        self.model = self._build_model(input_dim, prior_mean, prior_std)

        # Guide over GLOBALS ONLY — hide local 'theta'
        blocked = pyro.poutine.block(self.model, hide=["theta"])
        self.guide = AutoNormal(blocked, init_scale=0.1)

        if self.use_gpu:
            for _, v in self.guide.named_parameters():
                if v.device != self.device:
                    v.data = v.data.to(self.device)

        optimizer = pyro_optim.ClippedAdam({"lr": 5e-4, "weight_decay": weight_decay, "clip_norm": 5.0})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        # Training loop with mini-batching 
        n = Xtr.shape[0]
        losses = []
        for epoch in range(num_epochs):
            perm = torch.randperm(n, device=self.device)
            batch_losses = []
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                loss = svi.step(Xtr[idx], mu_tr[idx], sp_tr[idx], ytr[idx])
                batch_losses.append(loss / max(1, idx.numel()))
            epoch_loss = float(np.mean(batch_losses))
            if not np.isfinite(epoch_loss):
                print("[train] Detected non-finite loss. Stopping early.")
                break
            # Check guide params for NaN
            bad = False
            for name, p in self.guide.named_parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    print(f"[train] Non-finite in guide param: {name}. Stopping.")
                    bad = True
                    break
            if bad: break
            losses.append(epoch_loss)
            if verbose and ((epoch+1) % 100 == 0 or epoch == 0):
                print(f"[train] Epoch {epoch+1}/{num_epochs} - ELBO Loss: {epoch_loss:.4f}")

        # Posterior predictive over theta on test
        with torch.no_grad():
            predictive = Predictive(self.model, guide=self.guide, num_samples=1000, return_sites=["theta"])
            theta_samples = predictive(Xte, mu_te, sp_te, None)["theta"]   # [S, Nte]
            y_pred_mean = theta_samples.mean(dim=0).cpu().numpy()
            y_pred_std  = theta_samples.std(dim=0).cpu().numpy()
            y_pred_samples = theta_samples.cpu().numpy()                   # [S, Nte]

        # Compute metrics
        mae  = mean_absolute_error(y_test, y_pred_mean)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))

        self.train_summary = {
            'losses': losses,
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test,   'y_test': y_test,
            'y_pred_mean': y_pred_mean,
            'y_pred_std':  y_pred_std,
            'y_pred_samples': y_pred_samples,   # [S, N_test]
            'mae': mae, 'rmse': rmse,
            'prior_metrics': {k: {'mae': mean_absolute_error(y_test, prior_splits[k]),
                                  'rmse': np.sqrt(mean_squared_error(y_test, prior_splits[k]))}
                              for k in prior_splits},
            'prior_test': prior_splits,
            'feature_names': self.feature_names
        }

        print("\nTraining complete!")
        print(f"Test MAE: {mae:.4f}°, RMSE: {rmse:.4f}°")
        print("\nComparison with physics-based methods:")
        for k, m in self.train_summary['prior_metrics'].items():
            print(f"  {k.upper()}: MAE={m['mae']:.4f}°, RMSE={m['rmse']:.4f}°")

        return self.train_summary

    # -------------------------------------------------------- Inference ----------------------------------------------------- #
    def predict(self, X, mu_phys=None, sigma_phys=None, num_samples=1000, return_uncertainty=True, batch_size=512):
        """
        Make predictions using the trained hierarchical Bayesian AoA model.

        Parameters:
            - X                  [np.ndarray] : Feature matrix for prediction.
            - mu_phys            [np.ndarray] : Optional physics prior means for each sample.
            - sigma_phys         [np.ndarray] : Optional physics prior stddevs for each sample.
            - num_samples        [int]        : Number of posterior samples to draw.
            - return_uncertainty [bool]       : Whether to return uncertainty (stddev) along with mean predictions.
            - batch_size         [int]        : Batch size for prediction to manage memory usage.

        Returns:
            - mean               [np.ndarray] : Mean predictions.
            - std                [np.ndarray] : (Optional) Standard deviation of predictions.
        """
        # Verify that the model is created.
        if self.model is None:
            raise ValueError("Train the model before calling predict().")

        # Scale X
        Xs = np.zeros_like(X)
        for i in range(X.shape[1]):
            Xs[:, i] = self.scalers[i].transform(X[:, [i]]).ravel()
        Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)

        # Prepare physics side-channel if provided
        if mu_phys is None:
            mu_phys = np.zeros(X.shape[0], dtype=np.float32)
        if sigma_phys is None:
            sigma_phys = np.full(X.shape[0], fill_value=self.obs_sigma, dtype=np.float32)

        # Torch tensors for physics side-channel
        mu_t = torch.tensor(mu_phys,   dtype=torch.float32, device=self.device)
        sp_t = torch.tensor(sigma_phys,dtype=torch.float32, device=self.device)

        # Posterior predictive sampling in batches
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=["theta"])
        all_theta = []
        with torch.no_grad():
            for i in range(0, Xt.shape[0], batch_size):
                out = predictive(Xt[i:i+batch_size], mu_t[i:i+batch_size], sp_t[i:i+batch_size], None)
                all_theta.append(out["theta"].cpu().numpy())   # [S, B]
        theta = np.concatenate(all_theta, axis=1)  # [S, N]

        # Compute mean and uncertainty
        mean = theta.mean(axis=0)
        if return_uncertainty:
            std  = theta.std(axis=0)
            return mean, std
        return mean

    # -------------------------------------------------------- Visualization ------------------------------------------------- #
    def visualize_results(self, output_dir, experiment_name):
        """
        Visualize training results including predicted vs true angles, loss curves, error distributions, and posterior weight summaries.

        Parameters:
            - output_dir      [str] : Directory to save visualizations.
            - experiment_name [str] : Name of the experiment (used for subdirectory).
        
        Returns:
            - None
        """
        if self.train_summary is None:
            raise ValueError("Train the model first.")
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)

        # Extract training summary
        y_test = self.train_summary['y_test']
        y_pred_mean = self.train_summary['y_pred_mean']
        y_pred_std  = self.train_summary['y_pred_std']
        y_pred_samples = self.train_summary['y_pred_samples']  # [S, N]
        prior_test = self.train_summary['prior_test']
        losses = self.train_summary['losses']

        # Plot settings
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        # 1) Predicted vs True
        plt.figure(figsize=(10, 8))
        mn, mx = min(y_test.min(), y_pred_mean.min()), max(y_test.max(), y_pred_mean.max())
        plt.plot([mn, mx], [mn, mx], 'r--', label='Perfect Prediction')
        plt.errorbar(y_test, y_pred_mean, yerr=2*y_pred_std, fmt='o', alpha=0.8,
                     label='Bayesian Predictions (2$\\sigma$)', markersize=style.MARKER_SIZE)
        prior_colors = {'ds': 'g', 'weighted': 'm', 'music': 'c', 'phase': 'y'}
        prior_markers= {'ds': '^', 'weighted': 's', 'music': 'd', 'phase': 'x'}
        prior_labels = {'ds': 'DS', 'weighted': 'WDS', 'music': 'MUSIC', 'phase': 'PD'}
        for key in prior_test:
            plt.scatter(y_test, prior_test[key], marker=prior_markers[key],
                        color=prior_colors[key], alpha=0.35, s=70, label=f"{prior_labels[key]}")
        plt.xlabel('True Angle (degrees)')
        plt.ylabel('Predicted Angle (degrees)')
        plt.title(f'Hierarchical Bayesian AoA ({self.prior_type.upper()} prior)\n'
                  f'MAE: {self.train_summary["mae"]:.2f}°, RMSE: {self.train_summary["rmse"]:.2f}°')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "predicted_vs_true.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2) Loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('ELBO Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 3) Error distributions
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        errors = y_pred_mean - y_test
        plt.hist(errors, bins=20, alpha=0.7, label='Bayesian Model')
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Prediction Error (degrees)')
        plt.ylabel('Count')
        plt.title('Error Distribution (Model)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.subplot(2, 1, 2)
        for key in prior_test:
            pe = prior_test[key] - y_test
            plt.hist(pe, bins=20, alpha=0.5, label=key.upper())
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Prediction Error (degrees)')
        plt.ylabel('Count')
        plt.title('Error Distribution (Physics Methods)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "error_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 4) Posterior weight summaries from guide
        post_means, post_scales = {}, {}
        for name, param in self.guide.named_parameters():
            if 'AutoNormal.loc' in name and 'linear.weight' in name:
                post_means['weights'] = param.detach().cpu().numpy()
            elif 'AutoNormal.scale' in name and 'linear.weight' in name:
                post_scales['weights'] = np.abs(param.detach().cpu().numpy())
            elif 'AutoNormal.loc' in name and 'linear.bias' in name:
                post_means['bias'] = param.detach().cpu().numpy()
            elif 'AutoNormal.scale' in name and 'linear.bias' in name:
                post_scales['bias'] = np.abs(param.detach().cpu().numpy())
        if 'weights' in post_means:
            importance = np.abs(post_means['weights'][0])
            top_idx = np.argsort(importance)[-8:]
            plt.figure(figsize=(12, 10))
            for i, idx in enumerate(top_idx):
                feat = self.feature_names[idx] if idx < len(self.feature_names) else f'Feature {idx}'
                mu  = post_means['weights'][0, idx]
                sd  = post_scales['weights'][0, idx]
                x = np.linspace(mu - 3*sd, mu + 3*sd, 400)
                plt.subplot(4, 2, i+1)
                if sd > 0:
                    plt.plot(x, norm.pdf(x, mu, sd))
                    plt.fill_between(x, 0, norm.pdf(x, mu, sd), alpha=0.3)
                plt.axvline(0, color='r', linestyle='--', alpha=0.5)
                plt.axvline(mu, color='g', linestyle='-')
                plt.title(f'{feat} (\mu={mu:.3f}, \sigma={sd:.3f})')
                plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "weight_posteriors.png"), dpi=300, bbox_inches='tight')
            plt.close()

        # 5) Feature importance
        if 'weights' in post_means:
            imp = np.abs(post_means['weights'][0])
            imp = 100 * imp / max(imp.sum(), 1e-9)
            idx = np.argsort(imp)
            names = [self.feature_names[i] if i < len(self.feature_names) else f'F{i}' for i in idx]
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(names)), imp[idx], align='center')
            plt.yticks(range(len(names)), names)
            plt.xlabel('Relative Importance (%)')
            plt.title('Feature Importance (|w| normalized)')
            plt.grid(True, alpha=0.3)
            for i, v in enumerate(imp[idx]):
                plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
            plt.close()

        # 6) Uncertainty vs error
        plt.figure(figsize=(10, 8))
        abs_err = np.abs(y_pred_mean - y_test)
        plt.scatter(y_pred_std, abs_err, alpha=0.7, s=100)
        z = np.polyfit(y_pred_std, abs_err, 1)
        p = np.poly1d(z)
        xs = np.sort(y_pred_std)
        plt.plot(xs, p(xs), "r--", alpha=0.8, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        plt.xlabel('Prediction Uncertainty (std)')
        plt.ylabel('Absolute Error (degrees)')
        plt.title('Uncertainty vs Error')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "uncertainty_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        # Save a textual summary
        with open(os.path.join(vis_dir, "model_summary.txt"), 'w') as f:
            f.write("Hierarchical Bayesian AoA Model Summary\n")
            f.write("======================================\n\n")
            f.write(f"Config:\n  Prior Type: {self.prior_type}\n  Feature Mode: {self.feature_mode}\n"
                    f"  Features: {', '.join(self.feature_names)}\n  obs_sigma: {self.obs_sigma}\n\n")
            f.write(f"Performance:\n  MAE: {self.train_summary['mae']:.4f}°\n  RMSE: {self.train_summary['rmse']:.4f}°\n\n")
            f.write("Physics-based comparison:\n")
            for key, metrics in self.train_summary['prior_metrics'].items():
                f.write(f"  {key.upper()}: MAE={metrics['mae']:.4f}°, RMSE={metrics['rmse']:.4f}°\n")
        with open(os.path.join(vis_dir, "weights_bias_summary.txt"), 'w') as f:
            f.write("Posterior Weights and Bias (mean ± 2σ)\n")
            f.write("=====================================\n")
            if 'weights' in post_means:
                for i, feat in enumerate(self.feature_names):
                    mu = post_means['weights'][0, i]
                    sd = post_scales['weights'][0, i]
                    f.write(f"{feat:20s}: {mu:.4f} ± {2*sd:.4f}\n")
            if 'bias' in post_means:
                mu = post_means['bias'][0]
                sd = post_scales['bias'][0]
                f.write(f"\nBias: {mu:.4f} ± {2*sd:.4f}\n")

    def render_model_and_guide(self, output_dir, experiment_name):
        """
        Render and save the model structure, guide distributions, and graphical model.

        Parameters:
            - output_dir      [str] : Directory to save visualizations.
            - experiment_name [str] : Name of the experiment (used for subdirectory).

        Returns:
            - None
        """
        if self.model is None or self.guide is None:
            raise ValueError("Train the model first.")

        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)

        with open(os.path.join(vis_dir, "model_structure.txt"), 'w') as f:
            f.write(f"Model Structure:\n{str(self.model)}\n\n")
            f.write("Parameter Shapes:\n")
            for name, param in self.model.named_parameters():
                f.write(f"  {name}: {tuple(param.shape)}\n")

        # quick guide plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        weight_loc = weight_scale = bias_loc = bias_scale = None
        for name, p in self.guide.named_parameters():
            if 'weight' in name and 'loc' in name:   weight_loc  = p.detach().cpu().numpy()
            if 'weight' in name and 'scale' in name: weight_scale= np.abs(p.detach().cpu().numpy())
            if 'bias'   in name and 'loc' in name:   bias_loc    = p.detach().cpu().numpy()
            if 'bias'   in name and 'scale' in name: bias_scale  = np.abs(p.detach().cpu().numpy())

        if weight_loc is not None and weight_scale is not None:
            x = np.arange(weight_loc.shape[1])
            axes[0].errorbar(x, weight_loc[0], yerr=2*weight_scale[0], fmt='o', capsize=5)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(self.feature_names, rotation=90)
            axes[0].set_title(r'Weights (mean $\pm$ 2$\sigma$)')
            axes[0].grid(True, alpha=0.3)

        if bias_loc is not None and bias_scale is not None:
            axes[1].errorbar([0], bias_loc, yerr=2*bias_scale, fmt='o', capsize=5)
            axes[1].set_xticks([0]); axes[1].set_xticklabels(['Bias'])
            axes[1].set_title(r'Bias (mean $\pm$ 2$\sigma$)')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(vis_dir, "guide_distributions.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Graphical model
        plt.figure(figsize=(10, 8))
        plt.title("Hierarchical Bayesian AoA – Graphical Model")
        pos = {'weights': (0.25, 0.75),'bias': (0.25, 0.55),'tau': (0.25, 0.35),
               'mean': (0.55, 0.65),'theta': (0.75, 0.65),'phys': (0.85, 0.45),'y': (0.85, 0.75)}
        labels = {
            'weights':r'$w ~ N(\mu_p, \sigma_p)$','bias':r'$b ~ N(0,5)$','tau':r'$tau ~ HalfNormal(1)$',
            'mean':r'$\mu = Xw+b$','theta':r'$\theta_i ~ N(\mu_i,\tau)$','phys':r'$\mu_phys|\theta ~ N(\theta,\sigma_phys)$',
            'y':r'$y|\theta ~ N(\theta,\sigma_y)$'
        }
        edges = [('weights','mean'),('bias','mean'),('mean','theta'),('tau','theta'),
                 ('theta','y'),('theta','phys')]
        G = nx.DiGraph(); G.add_nodes_from(pos); G.add_edges_from(edges)
        nx.draw_networkx_nodes(G,pos,node_size=3000,node_color='lightblue',alpha=0.8)
        nx.draw_networkx_edges(G,pos,width=2,arrowsize=20)
        nx.draw_networkx_labels(G,{k:(v[0],v[1]-0.02) for k,v in pos.items()},
                                labels=labels,font_size=style.ANNOTATION_SIZE,
                                bbox=dict(facecolor='white',alpha=0.7,boxstyle='round,pad=0.5'))
        plt.axis('off'); plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "graphical_model.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_prior_vs_posterior(self, output_dir, experiment_name, num_samples=5):
        """
        Visualize prior vs posterior distributions for a subset of test samples.

        Parameters:
            - output_dir      [str] : Directory to save visualizations.
            - experiment_name [str] : Name of the experiment (used for subdirectory).
            - num_samples     [int] : Number of test samples to visualize.

        Returns:
            - None
        """
        if self.train_summary is None:
            raise ValueError("Train first.")
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)

        y_test = self.train_summary['y_test']
        y_pred_mean = self.train_summary['y_pred_mean']
        y_pred_std  = self.train_summary['y_pred_std']
        prior_test  = self.train_summary['prior_test']

        if self.prior_type in self.train_summary['prior_metrics']:
            prior_std = max(0.1, self.train_summary['prior_metrics'][self.prior_type]['mae'])
        else:
            prior_std = 1.0

        if len(y_test) <= num_samples:
            idxs = np.arange(len(y_test))
        else:
            srt = np.argsort(y_test)
            step= max(1, len(srt)//num_samples)
            idxs = srt[::step][:num_samples]

        fig, axes = plt.subplots(len(idxs), 1, figsize=(10, 3*len(idxs)))
        if len(idxs) == 1: axes = [axes]
        for i, idx in enumerate(idxs):
            ax = axes[i]
            true_angle = y_test[idx]
            post_mu, post_sd = y_pred_mean[idx], y_pred_std[idx]
            prior_mu = prior_test[self.prior_type][idx] if self.prior_type in prior_test else 0.0
            rng = 4*max(prior_std, post_sd)
            x = np.linspace(min(prior_mu, post_mu)-rng, max(prior_mu, post_mu)+rng, 800)
            ax.plot(x, norm.pdf(x, prior_mu, prior_std), 'r--', lw=2, label=f'{self.prior_type.upper()} Prior')
            ax.fill_between(x, 0, norm.pdf(x, prior_mu, prior_std), color='red', alpha=0.2)
            ax.plot(x, norm.pdf(x, post_mu, post_sd), 'b-', lw=2, label='Posterior')
            ax.fill_between(x, 0, norm.pdf(x, post_mu, post_sd), color='blue', alpha=0.2)
            ax.axvline(true_angle, color='k', lw=2, label='True')
            ax.set_xlabel('Angle (°)'); ax.set_ylabel('Density')
            ax.set_title(f'Sample {i+1}: Prior vs Posterior (True: {true_angle:.2f}°)')
            ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(vis_dir, "prior_vs_posterior.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)


    def plot_posterior_predictive(self, output_dir, experiment_name):
        """
        Plot the posterior predictive distribution over the test set.

        Parameters:
            - output_dir      [str] : Directory to save visualizations.
            - experiment_name [str] : Name of the experiment (used for subdirectory).

        Returns:
            - None
        """
        if self.train_summary is None:
            raise ValueError("Train first.")
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        y_test = self.train_summary['y_test']
        S, N = self.train_summary['y_pred_samples'].shape
        # Convert to a band plot
        yp = self.train_summary['y_pred_samples']  # [S, N]
        sort_idx = np.argsort(y_test)
        y_test_sorted = y_test[sort_idx]
        y5, y25, y50, y75, y95 = np.percentile(yp[:, sort_idx], [5,25,50,75,95], axis=0)
        print("Interval widths:", np.mean(y95 - y5), np.mean(y75 - y25))
        plt.figure(figsize=(12, 8))
        plt.fill_between(range(N), y5, y95, alpha=0.2, color='skyblue', label='90% CI')
        plt.fill_between(range(N), y25, y75, alpha=0.4, color='dodgerblue', label='50% CI')
        plt.plot(range(N), y50, 'black', lw=2, label='Median')
        plt.plot(range(N), y_test_sorted, 'ro', ms=8, label='True')
        plt.xlabel('Test Point (sorted by true angle)')
        plt.ylabel('Angle (°)')
        plt.title('Posterior Predictive Distribution')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "posterior_predictive.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_uncertainty_calibration(self, output_dir, experiment_name):
        """
        Plot uncertainty calibration using standardized errors and KS test.

        Parameters:
            - output_dir      [str] : Directory to save visualizations.
            - experiment_name [str] : Name of the experiment (used for subdirectory).

        Returns:
            - None
        """
        if self.train_summary is None:
            raise ValueError("Train first.")
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        y_test = self.train_summary['y_test']
        mu = self.train_summary['y_pred_mean']
        sd = self.train_summary['y_pred_std']
        z = (y_test - mu) / np.maximum(sd, 1e-9)
        plt.figure(figsize=(10, 8))
        plt.hist(z, bins=20, density=True, alpha=0.6, label='Standardized Errors')
        x = np.linspace(-4, 4, 1000)
        plt.plot(x, norm.pdf(x), 'r-', lw=2, label='Standard Normal')
        plt.xlabel('z-score')
        plt.ylabel('Density')
        plt.title('Uncertainty Calibration')
        plt.legend(); plt.grid(True, alpha=0.3)
        ks_stat, ks_p = kstest(z, 'norm')
        plt.text(0.05, 0.95, f'KS: stat={ks_stat:.3f}, p={ks_p:.3f}',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "uncertainty_calibration.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------- Optional analysis using posterior weights (kept from baseline) ---------------- #
    def analyze_with_posterior_weights(self, data_manager, output_dir, experiment_name):
        """
        Analyze selected samples using the posterior weight mean to create a weighted spectrum.

        Parameters:
            - data_manager    [DataManager] : DataManager instance with signal and metadata.
            - output_dir      [str]         : Directory to save analysis results.
            - experiment_name [str]         : Name of the experiment (used for subdirectory).
        
        Returns:
            - None
        """
        if self.model is None:
            raise ValueError("Train first.")
        analysis_dir = os.path.join(output_dir, "bayesian_model", experiment_name, "posterior_weighted_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        # Extract posterior weight mean
        weight_loc = None
        for name, p in self.guide.named_parameters():
            if 'weight' in name and 'loc' in name:
                weight_loc = p.detach().cpu().numpy()
        if weight_loc is None:
            raise ValueError("Could not extract posterior weights")
        if data_manager.results is None:
            data_manager.analyze_all_data(save_results=False)
        test_indices = np.linspace(0, len(data_manager.metadata)-1, 5, dtype=int)
        for idx in test_indices:
            meta = data_manager.metadata.iloc[idx]
            signals = data_manager.signal_data[idx]
            D, W, L, wavelength = meta['D'], meta['W'], meta['L'], meta['lambda']
            true_angle = data_manager.get_true_angle(D, W)
            phasor1, phasor2 = signals['phasor1'], signals['phasor2']
            rssi1, rssi2 = signals['rssi1'], signals['rssi2']
            analysis_aoa = np.arange(main.MIN_ANGLE, main.MAX_ANGLE + main.STEP, main.STEP)
            original = main.analyze_aoa(phasor1, phasor2, rssi1, rssi2, L, wavelength, analysis_aoa, true_angle)
            feats = self._extract_features_for_sample(phasor1, phasor2, rssi1, rssi2, D, W, wavelength)
            scaled = np.zeros_like(feats)
            for i in range(feats.shape[0]):
                scaled[i] = self.scalers[i].transform([[feats[i]]])[0][0]
            X_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model.linear(X_tensor).squeeze(-1).item()
            bayesian_aoa = np.arange(main.BAYESIAN_MIN_ANGLE, main.BAYESIAN_MAX_ANGLE + main.STEP, main.STEP)
            batch_size = len(bayesian_aoa)
            feature_batch = np.tile(scaled, (batch_size, 1))
            batch_tensor = torch.tensor(feature_batch, dtype=torch.float32, device=self.device)
            weighted_spectrum = np.zeros(batch_size)
            chunk = 128
            for s in range(0, batch_size, chunk):
                e = min(s + chunk, batch_size)
                with torch.no_grad():
                    out = self.model.linear(batch_tensor[s:e]).squeeze(-1).cpu().numpy()
                    weighted_spectrum[s:e] = out
            weighted_spectrum = np.exp(-(weighted_spectrum - pred)**2)
            weighted_spectrum /= max(weighted_spectrum.max(), 1e-9)
            weighted_angle = bayesian_aoa[np.argmax(weighted_spectrum)]
            # plots
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(analysis_aoa, original['spectra']['ds'], 'r-', label='Original DS')
            plt.plot(analysis_aoa, original['spectra']['weighted'], 'm--', label='Original Weighted')
            full = np.zeros_like(analysis_aoa, dtype=float)
            i0 = np.searchsorted(analysis_aoa, main.BAYESIAN_MIN_ANGLE)
            i1 = np.searchsorted(analysis_aoa, main.BAYESIAN_MAX_ANGLE)
            rng = i1 - i0
            for i in range(i0, i1):
                j = int((i - i0) * (len(bayesian_aoa) / max(rng,1)))
                if j < len(weighted_spectrum):
                    full[i] = weighted_spectrum[j]
            plt.plot(analysis_aoa, full, 'g-', lw=2, label='Bayesian Weighted (±15°)')
            plt.axvspan(main.BAYESIAN_MIN_ANGLE, main.BAYESIAN_MAX_ANGLE, color='lightgreen', alpha=0.2, label='Bayesian Range')
            plt.axvline(original['angles']['ds'], color='r', ls=':', label='DS Est.')
            plt.axvline(original['angles']['weighted'], color='m', ls=':', label='Weighted Est.')
            plt.axvline(weighted_angle, color='g', ls=':', label='Bayesian Est.')
            plt.axvline(true_angle, color='k', ls='-', label='True')
            plt.xlabel('Angle (°)'); plt.ylabel('Normalized Power')
            plt.title(f'Spectrum Comparison (D={D:.2f}m, W={W:.2f}m)')
            plt.grid(True, alpha=0.3); plt.legend()
            plt.subplot(2, 1, 2)
            methods = ['Phase','DS','Weighted','MUSIC','Bayesian']
            errors = [
                abs(original['angles']['phase']   - true_angle),
                abs(original['angles']['ds']      - true_angle),
                abs(original['angles']['weighted']- true_angle),
                abs(original['angles']['music']   - true_angle),
                abs(weighted_angle                - true_angle)
            ]
            plt.bar(methods, errors)
            plt.ylabel('Abs. Error (°)'); plt.title('Error Comparison')
            plt.grid(True, alpha=0.3)
            for i, v in enumerate(errors):
                plt.text(i, v + 0.1, f'{v:.2f}°', ha='center')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f'posterior_weighted_D{D:.2f}_W{W:.2f}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

    def _extract_features_for_sample(self, phasor1, phasor2, rssi1, rssi2, D, W, wavelength):
        """
        Extract features for a single sample based on the specified feature mode.

        Parameters:
            - phasor1    [np.ndarray] : Phasor data from antenna 1.
            - phasor2    [np.ndarray] : Phasor data from antenna 2.
            - rssi1      [np.ndarray] : RSSI data from antenna 1.
            - rssi2      [np.ndarray] : RSSI data from antenna 2.
            - D          [float]      : Distance between antennas.
            - W          [float]      : Width of the antenna array.
            - wavelength [float]      : Wavelength of the signal.

        Returns:
            - features   [np.ndarray] : Extracted feature vector.
        """
        phase1_mean = np.angle(np.mean(phasor1))
        phase2_mean = np.angle(np.mean(phasor2))
        phase_diff  = np.angle(np.exp(1j * (phase1_mean - phase2_mean)))
        mag1_mean   = np.mean(np.abs(phasor1))
        mag2_mean   = np.mean(np.abs(phasor2))
        rssi1_mean  = np.mean(rssi1)
        rssi2_mean  = np.mean(rssi2)
        rssi_diff   = rssi1_mean - rssi2_mean
        ph1_r, ph1_i = np.mean(phasor1.real), np.mean(phasor1.imag)
        ph2_r, ph2_i = np.mean(phasor2.real), np.mean(phasor2.imag)
        features = np.array([
            phase1_mean, phase2_mean, phase_diff,
            mag1_mean, mag2_mean,
            rssi1_mean, rssi2_mean, rssi_diff,
            ph1_r, ph1_i, ph2_r, ph2_i,
            wavelength
        ])
        if self.feature_mode in ['full', 'width_only']:
            features = np.append(features, W)
        if self.feature_mode == 'full':
            features = np.append(features, D)
        return features
    
def compare_bayesian_models(results_dict, output_dir, experiment_name):
    """
    Compare Bayesian model variants with bar charts of MAE and RMSE.
    
    Parameters:
        - results_dict    [dict] : Dictionary with model names as keys and their performance summaries as values.
        - output_dir      [str]  : Directory to save the comparison plots.
        - experiment_name [str]  : Name of the experiment for organizing output.

    Returns:
        - Saves a bar chart comparing MAE and RMSE of different models.
    """
    # Directory and filename setup
    vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
    os.makedirs(vis_dir, exist_ok=True)
    # Extract model names and their MAE/RMSE
    model_names = []
    maes, rmses = [], []
    for name, summary in results_dict.items():
        if 'mae' in summary and 'rmse' in summary:
            model_names.append(name)
            maes.append(summary['mae'])
            rmses.append(summary['rmse'])
    x = np.arange(len(model_names))
    # Plotting
    width = 0.35
    plt.figure(figsize=(14, 6))
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.bar(x - width/2, maes, width, label='MAE')
    plt.bar(x + width/2, rmses, width, label='RMSE')
    plt.xticks(x, model_names, rotation=45)
    plt.ylabel('Error (degrees)')
    plt.title('Bayesian Model Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "bayesian_model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def figure1_model_comparison(results_dict, output_dir, experiment_name):
    """
    Create a bar chart comparing different models based on their MAE and RMSE.
    Configuration is special, as the figures are for the ICASSP conference paper.

    Parameters:
        - results_dict    [dict] : Dictionary with model names as keys and their performance summaries as values.
        - output_dir      [str]  : Directory to save the comparison plots.
        - experiment_name [str]  : Name of the experiment for organizing output.

    Returns: 
        - Saves a bar chart comparing MAE and RMSE of different models.
    """
    # Increase default font sizes for paper readability
    rcParams['font.size'] = 20
    rcParams['axes.titlesize'] = 22
    rcParams['axes.labelsize'] = 20
    rcParams['xtick.labelsize'] = 19
    rcParams['ytick.labelsize'] = 19
    rcParams['legend.fontsize'] = 19
    # Directory setup
    vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
    os.makedirs(vis_dir, exist_ok=True)
    # Extract model names and their MAE/RMSE
    names, maes, rmses = [], [], []
    for name, s in results_dict.items():
        if isinstance(s, dict) and ('mae' in s and 'rmse' in s):
            names.append(name); maes.append(float(s['mae'])); rmses.append(float(s['rmse']))
    # Sort by RMSE
    order = np.argsort(rmses)
    names  = [names[i]  for i in order]
    maes   = [maes[i]   for i in order]
    rmses  = [rmses[i]  for i in order]
    # Plotting
    x = np.arange(len(names)); w = 0.42
    mae_color  = "#0A2342"   # deep navy
    rmse_color = "#17BECF"   # cyan
    fig, ax    = plt.subplots(figsize=(14, 6))
    bars_mae   = ax.bar(x - w/2, maes,  width=w, color=mae_color,  label='MAE')
    bars_rmse  = ax.bar(x + w/2, rmses, width=w, color=rmse_color, label='RMSE', alpha=0.95)
    ax.set_xticks(x, names, rotation=45, ha='right')
    ax.set_ylabel('Error (°)')
    ax.set_title('Model Comparison — MAE vs RMSE')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper left')
    # Helper Function: place 4-decimal point labels vertically inside bars
    def _inside_labels(bars, vals, *, color_mode="auto"):
        for b, v in zip(bars, vals):
            h = b.get_height()
            txt_color = "black" if color_mode == "black" else ("white" if h >= 0.25 else "black")
            ax.text(b.get_x() + b.get_width()/2, h/2,
                    f"{v:.4f}°", ha='center', va='center',
                    rotation=90, fontsize=16, color=txt_color)

    _inside_labels(bars_mae, maes,  color_mode="auto")   # auto white/black
    _inside_labels(bars_rmse, rmses, color_mode="black") # force black on cyan
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "ICASSP_Fig1_model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def figure2_scatter_and_posterior(best_summary, output_dir, experiment_name, model_label="Best model", magnify=25):
    """
    Create a figure with two subplots: (1) Scatter plot of predicted vs true AoA with uncertainty,
    and (2) Posterior predictive distribution with uncertainty bands.
    Configuration is special, as the figures are for the ICASSP conference paper.

    Parameters:
        - best_summary    [dict] : Summary dictionary from the best Bayesian model containing predictions and uncertainties.
        - output_dir      [str]  : Directory to save the plots.
        - experiment_name [str]  : Name of the experiment for organizing output.
        - model_label     [str]  : Label for the model to be used in titles/legends.
        - magnify         [int]  : Factor to magnify uncertainty bands in posterior plot for visibility.

    Returns:
        - Saves a figure with scatter plot and posterior predictive distribution.
    """
    # Increase default font sizes for paper readability
    rcParams['font.size'] = 20
    rcParams['axes.titlesize'] = 22
    rcParams['axes.labelsize'] = 20
    rcParams['xtick.labelsize'] = 19
    rcParams['ytick.labelsize'] = 19
    rcParams['legend.fontsize'] = 16
    # Directory setup
    vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
    os.makedirs(vis_dir, exist_ok=True)
    # Extract data from summary
    y_true  = np.asarray(best_summary['y_test'])
    mu      = np.asarray(best_summary['y_pred_mean'])
    std     = np.asarray(best_summary['y_pred_std'])
    samples = np.asarray(best_summary['y_pred_samples'])  # [S, N]
    priors  = best_summary.get('prior_test', {})
    # Helper functions to create each plot
    def create_scatter_plot(ax):
        mn, mx = float(min(y_true.min(), mu.min())), float(max(y_true.max(), mu.max()))
        ax.plot([mn, mx], [mn, mx], linestyle='--', linewidth=1.2, color='r', label='Perfect Prediction')
        # Same palette as visualize_results()
        shape_map = {'ds': '^', 'weighted': 'x', 'music': 's', 'phase': 'd'}
        color_map = {'ds': 'g',  'weighted': 'm', 'music': 'c', 'phase': 'y'}
        for key, vals in priors.items():
            mk = shape_map.get(key, 'o')
            col = color_map.get(key, '#888888')
            if mk == 'x':
                ax.scatter(y_true, np.asarray(vals), marker=mk, color=col, s=58, alpha=0.7, label=key.upper())
            else:
                ax.scatter(y_true, np.asarray(vals), marker=mk, color=col, s=58, alpha=0.7,
                           edgecolor='none', label=key.upper())
        ax.errorbar(y_true, mu, yerr=2*std, fmt='o', markersize=8,
                    alpha=0.95, label='Bayesian Predictions (2$\\sigma$)')
        ax.set_xlabel('True AoA (°)')
        ax.set_ylabel('Predicted AoA (°)')
        ax.set_title(f'Predicted vs True — {model_label}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        return ax
    def create_posterior_plot(ax):
        order = np.argsort(y_true)
        y_sorted = y_true[order]
        q5, q25, q50, q75, q95 = np.percentile(samples[:, order], [5, 25, 50, 75, 95], axis=0)
        q5m  = q50 - (q50 - q5)  * magnify
        q25m = q50 - (q50 - q25) * magnify
        q75m = q50 + (q75 - q50) * magnify
        q95m = q50 + (q95 - q50) * magnify
        xs = np.arange(len(y_sorted))
        true_red      = "#D7263D"
        median_black  = "#000000"
        band50_dark   = "#0A2342"  # 50%
        band90_cyan   = "#5E88C6"  # 90%
        ax.fill_between(xs, q5m,  q95m,  alpha=0.30, color=band90_cyan, label='90% band')
        ax.fill_between(xs, q25m, q75m,  alpha=0.45, color=band50_dark, label='50% band')
        ax.plot(xs, q50, linewidth=1.9, color=median_black, label='Median')
        ax.plot(xs, y_sorted, 'o', markersize=4.5, color=true_red, label='True')
        ax.set_xlabel('Test sample (sorted by true AoA)')
        ax.set_ylabel('AoA (°)')
        ax.set_title('Posterior Predictive (bands magnified for visibility)')
        ax.grid(True, alpha=0.3)
        leg = ax.legend(loc='best')
        if leg is not None:
            leg.set_title(f"Magnification: ×{magnify}", prop={'size': 16})
        return ax
    # Create the combined figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    create_scatter_plot(axes[0])
    create_posterior_plot(axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "ICASSP_Fig2_scatter_and_posterior.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    # Figure 1: Scatter plot
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    create_scatter_plot(ax1)
    fig1.tight_layout()
    fig1.savefig(os.path.join(vis_dir, "ICASSP_Fig2a_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    # Figure 2: Posterior plot
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    create_posterior_plot(ax2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(vis_dir, "ICASSP_Fig2b_posterior.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)


# =================================================================================================================================== #
# --------------------------------------------------------- ORCHESTRATION ----------------------------------------------------------- #
def train_bayesian_models(data_manager, results_dir, num_epochs=2000):
    """
    Train and evaluate multiple Bayesian AoA regression models with different priors and feature modes.

    Parameters:
        - data_manager [DataManager] : Instance of DataManager containing training and test data.
        - results_dir  [str]         : Directory to save results and visualizations.
        - num_epochs   [int]         : Number of training epochs for each model.
    
    Returns:
        - Dictionary containing:
            - "results": All trained models with their performance summaries.
            - "best_name": Name of the best-performing model based on RMSE.
            - "best": The best model's entry (model instance and summary).
    """
    # Verify results directory exists
    os.makedirs(results_dir, exist_ok=True)
    # Define configurations to try
    configs = []
    for prior in ['ds','weighted','music','phase']:
        for fmode in ['full','width_only','sensor_only']:
            configs.append((prior, fmode))
    # Train and evaluate each configuration
    models = {}
    for prior, fmode in configs:
        exp_name = f"{prior}_{fmode}"
        print(f"\n=== Training {exp_name} ===")
        model = BayesianAoARegressor(prior_type=prior, feature_mode=fmode, obs_sigma=1e-2)
        summary = model.train(data_manager, num_epochs=num_epochs, learning_rate=1e-3, batch_size=128, verbose=True)
        model.visualize_results(results_dir, exp_name)
        model.render_model_and_guide(results_dir, exp_name)
        model.visualize_prior_vs_posterior(results_dir, exp_name, num_samples=5)
        model.plot_posterior_predictive(results_dir, exp_name)
        model.plot_uncertainty_calibration(results_dir, exp_name)
        models[exp_name] = {'model': model, 'summary': summary}
    # Compare models and obtain the best one
    best_name = min(models, key=lambda n: models[n]['summary']['rmse'])
    best_model = models[best_name]
    # Return results
    return {
        "results": models,       # all models with summaries
        "best_name": best_name,  # name of the best model
        "best": best_model       # the best model entry
    }
# =================================================================================================================================== #