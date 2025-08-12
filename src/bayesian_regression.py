# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import src.data_management as dm  # Data management functions for organizing and analyzing RFID data.                                 #
import main as main # Main module for running the analysis and managing experiments.                                                  #
import os                                            # Operating system interfaces for file and directory manipulation.               #
import numpy as np                                   # Mathematical functions.                                                        #
import matplotlib.pyplot as plt                      # Data visualization.                                                            #
from   scipy.stats import norm                       # Statistical functions for normal distribution fitting.                         #
import torch                                         # PyTorch for deep learning and tensor operations.                               #
import torch.optim as optim                          # Optimization algorithms for training models.                                   #
from sklearn.preprocessing import StandardScaler     # Standardization of features by removing the mean and scaling to unit variance. #
from sklearn.model_selection import train_test_split # Train-test split for model evaluation.                                         #
from sklearn.metrics import mean_absolute_error      # Mean absolute error for regression tasks.                                      #
from sklearn.metrics import mean_squared_error       # Mean squared error for regression tasks.                                       #
import pyro                                          # Pyro for probabilistic programming and Bayesian inference.                     #
import pyro.distributions as dist                    # Pyro distributions for probabilistic modeling.                                 #
from pyro.nn import PyroModule, PyroSample           # PyroModule for creating probabilistic models.                                  #
from pyro.infer import SVI, Trace_ELBO, Predictive   # Stochastic Variational Inference (SVI) for training models.                    #
from pyro.infer.autoguide import AutoNormal          # AutoGuide for automatic guide generation in Pyro.                              #
import pyro.optim as optim                           # Pyro optimization algorithms for training models.                              #
from scipy.stats import norm, kstest                 # Statistical functions for normal distribution fitting and fit tests.           #
import networkx as nx                                # NetworkX for graph-based operations.                                           #
# =================================================================================================================================== #


# =================================================================================================================================== #
# ----------------------------------------------------- MACHINE LEARNING ANALYSIS --------------------------------------------------- #
# Set seeds for reproducibility
pyro.set_rng_seed(42)
torch.manual_seed(42)
np.random.seed(42)

class BayesianAoARegressor:
    """
    Bayesian Angle of Arrival (AoA) Regressor that incorporates physics-based priors
    from MUSIC and beamforming methods.
    
    This model uses Pyro for Bayesian inference and allows for different types of priors
    based on the physical understanding of the AoA estimation problem.
    """
    
    def __init__(self, use_gpu=True, prior_type='ds', feature_mode='full'):
        """
        Initialize the Bayesian AoA Regressor.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
            prior_type (str): Type of prior to use ('ds', 'music', 'weighted', 'flat')
            feature_mode (str): Feature set to use ('full', 'width_only', 'sensor_only')
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.prior_type = prior_type
        self.feature_mode = feature_mode
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
    
    def _extract_features(self, data_manager, include_distance=True, include_width=True):
        """Extract features from data manager for Bayesian regression"""
        # Verify that results are available
        if data_manager.results is None:
            data_manager.analyze_all_data(save_results=False)

        # Prepare features
        all_features = []
        all_angles = []
        prior_estimates = {
            'ds': [],
            'weighted': [],
            'music': [],
            'phase': []
        }
        
        # Track feature names
        feature_names = []

        for idx, meta in enumerate(data_manager.metadata.iterrows()):
            meta = meta[1]  # Get the actual Series from the tuple

            # Extract signal data
            signals = data_manager.signal_data[idx]
            phasor1 = signals['phasor1']
            phasor2 = signals['phasor2']
            rssi1 = signals['rssi1']
            rssi2 = signals['rssi2']

            # Calculate true angle
            true_angle = data_manager.get_true_angle(meta['D'], meta['W'])

            # Store prior estimates from results dataframe
            result_row = data_manager.results.iloc[idx]
            prior_estimates['ds'].append(result_row['theta_ds'])
            prior_estimates['weighted'].append(result_row['theta_weighted'])
            prior_estimates['music'].append(result_row['theta_music'])
            prior_estimates['phase'].append(result_row['theta_phase'])

            # Feature Extraction
            features = []
            
            # Mean phase and magnitude values
            phase1_mean = np.angle(np.mean(phasor1))
            phase2_mean = np.angle(np.mean(phasor2))
            mag1_mean = np.mean(np.abs(phasor1))
            mag2_mean = np.mean(np.abs(phasor2))
            
            # Phase difference
            phase_diff = phase1_mean - phase2_mean
            phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap to [-π, π]
            
            # RSSI features
            rssi1_mean = np.mean(rssi1)
            rssi2_mean = np.mean(rssi2)
            rssi_diff = rssi1_mean - rssi2_mean
            
            # Phasor correlations (using real and imaginary parts)
            phasor1_real_mean = np.mean(phasor1.real)
            phasor1_imag_mean = np.mean(phasor1.imag)
            phasor2_real_mean = np.mean(phasor2.real)
            phasor2_imag_mean = np.mean(phasor2.imag)
            
            # Wavelength
            wavelength = meta['lambda']
            
            # Add basic phasor-derived features (always included)
            if len(feature_names) == 0:  # Only add names once
                feature_names.extend([
                    'phase1_mean', 'phase2_mean', 'phase_diff',
                    'mag1_mean', 'mag2_mean',
                    'rssi1_mean', 'rssi2_mean', 'rssi_diff',
                    'phasor1_real_mean', 'phasor1_imag_mean',
                    'phasor2_real_mean', 'phasor2_imag_mean',
                    'wavelength'
                ])
            
            features.extend([
                phase1_mean, phase2_mean, phase_diff,
                mag1_mean, mag2_mean,
                rssi1_mean, rssi2_mean, rssi_diff,
                phasor1_real_mean, phasor1_imag_mean,
                phasor2_real_mean, phasor2_imag_mean,
                wavelength
            ])
            
            # Add geometric features if specified
            if include_distance:
                distance = meta['D']
                features.append(distance)
                if len(feature_names) == 13:  # Add name only once
                    feature_names.append('distance')
            
            if include_width:
                width = meta['W']
                features.append(width)
                # Add name only if not already added
                if (include_distance and len(feature_names) == 14) or \
                (not include_distance and len(feature_names) == 13):
                    feature_names.append('width')

            all_features.append(features)
            all_angles.append(true_angle)

        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_angles)
        
        # Convert prior estimates to numpy arrays
        for key in prior_estimates:
            prior_estimates[key] = np.array(prior_estimates[key])

        self.feature_names = feature_names
        return X, y, feature_names, prior_estimates
    
    def _build_model(self, input_dim, prior_mean, prior_std):
        """
        Build the Bayesian regression model using Pyro.
        
        Args:
            input_dim (int): Number of input features
            prior_mean (float): Mean for the weight prior
            prior_std (float): Standard deviation for the weight prior
        """
        class BayesianLinearRegression(PyroModule):
            def __init__(self, input_dim, prior_mean=0.0, prior_std=1.0, device='cpu'):
                super().__init__()
                self.device = device
                
                # Register the weights as a PyroSample
                self.linear = PyroModule[torch.nn.Linear](input_dim, 1).to(device)
                
                # Set informative priors for weights based on physics
                weight_prior_mean = torch.ones(1, input_dim, device=device) * prior_mean
                weight_prior_std = torch.ones(1, input_dim, device=device) * prior_std
                
                self.linear.weight = PyroSample(
                    dist.Normal(weight_prior_mean, weight_prior_std).to_event(2)
                )
                
                # Prior for bias with explicit device placement
                bias_prior = torch.zeros(1, device=device)
                bias_scale = torch.ones(1, device=device)
                self.linear.bias = PyroSample(
                    dist.Normal(bias_prior, bias_scale).to_event(1)
                )
                
            def forward(self, x, y=None):
                # Ensure x is on the correct device
                if x.device != self.device:
                    x = x.to(self.device)
                    
                # Get predicted angle from linear model
                mean = self.linear(x).squeeze(-1)
                
                # Observation noise (learnable) with explicit device placement
                sigma = pyro.sample("sigma", dist.LogNormal(
                    torch.tensor(0.0, device=self.device), 
                    torch.tensor(1.0, device=self.device)
                ))
                
                # Condition on observed data if provided
                with pyro.plate("data", x.shape[0]):
                    # Ensure y is on the correct device if provided
                    if y is not None and y.device != self.device:
                        y = y.to(self.device)
                    obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
                
                return mean
        
        # Create model and move to the correct device
        model = BayesianLinearRegression(input_dim, prior_mean, prior_std, device=self.device)
        
        # Double-check all parameters are on the correct device
        for name, param in model.named_parameters():
            if param.device != self.device:
                param.data = param.data.to(self.device)
                
        return model
    
    def train(self, data_manager, num_epochs=1000, learning_rate=0.01):
        """
        Train the Bayesian AoA regression model.
        
        Args:
            data_manager: The DataManager object containing the dataset
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate for the optimizer
            
        Returns:
            dict: Training metrics and summary
        """
        # Determine feature inclusion based on feature_mode
        include_distance = self.feature_mode == 'full'
        include_width = self.feature_mode in ['full', 'width_only']
        
        # Extract features and data
        X, y, feature_names, prior_estimates = self._extract_features(
            data_manager, include_distance, include_width)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42)
        
        # Also split prior estimates for visualization
        prior_splits = {}
        for key in prior_estimates:
            _, prior_test = train_test_split(
                prior_estimates[key], test_size=0.1, random_state=42)
            prior_splits[key] = prior_test
        
        # Standardize features
        self.scalers = []
        X_train_scaled = np.zeros_like(X_train)
        X_test_scaled = np.zeros_like(X_test)
        
        for i in range(X_train.shape[1]):
            scaler = StandardScaler()
            X_train_scaled[:, i] = scaler.fit_transform(X_train[:, i].reshape(-1, 1)).flatten()
            X_test_scaled[:, i] = scaler.transform(X_test[:, i].reshape(-1, 1)).flatten()
            self.scalers.append(scaler)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        
        # Determine prior parameters based on prior_type
        prior_mean = 0.0
        prior_std = 1.0
        
        if self.prior_type in prior_estimates:
            # Calculate the standard deviation of the difference between
            # the physics-based estimate and the true angle
            prior_error = prior_estimates[self.prior_type] - y
            prior_std = max(0.1, np.std(prior_error))  # Ensure minimum variance
            
            print(f"Using {self.prior_type} prior with std={prior_std:.4f}")
        
        # Clear any existing parameters
        pyro.clear_param_store()
        
        # Create the model and guide
        input_dim = X_train.shape[1]
        self.model = self._build_model(input_dim, prior_mean, prior_std)

        # Create guide with explicit device placement
        self.guide = AutoNormal(self.model)

        # Explicitly move all guide parameters to the correct device
        if self.use_gpu:
            for name, value in self.guide.named_parameters():
                if value.device != self.device:
                    value.data = value.data.to(self.device)
        
        # Setup SVI (Stochastic Variational Inference)
        optimizer = optim.Adam({"lr": learning_rate})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        # Training loop
        losses = []
        for epoch in range(num_epochs):
            loss = svi.step(X_train_tensor, y_train_tensor)
            losses.append(loss)
            
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}")
        
        # Evaluate on test set
        self.model.eval()
        
        # Perform predictions with uncertainty
        predictive = Predictive(self.model, guide=self.guide, num_samples=1000)
        samples = predictive(X_test_tensor)
        
        # Extract predictions and calculate mean and std
        y_pred_samples = samples["obs"].cpu().numpy()
        y_pred_mean = y_pred_samples.mean(axis=0)
        y_pred_std = y_pred_samples.std(axis=0)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred_mean)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
        
        # Also compute metrics for the prior methods on test set
        prior_metrics = {}
        for key in prior_splits:
            prior_mae = mean_absolute_error(y_test, prior_splits[key])
            prior_rmse = np.sqrt(mean_squared_error(y_test, prior_splits[key]))
            prior_metrics[key] = {
                'mae': prior_mae,
                'rmse': prior_rmse
            }
        
        # Store training summary
        self.train_summary = {
            'losses': losses,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_mean': y_pred_mean,
            'y_pred_std': y_pred_std,
            'y_pred_samples': y_pred_samples,
            'mae': mae,
            'rmse': rmse,
            'prior_metrics': prior_metrics,
            'prior_test': prior_splits,
            'feature_names': feature_names
        }
        
        print(f"\nTraining complete!")
        print(f"Test MAE: {mae:.4f}°, RMSE: {rmse:.4f}°")
        
        # Print comparison with prior methods
        print("\nComparison with physics-based methods:")
        for key, metrics in prior_metrics.items():
            print(f"  {key.upper()}: MAE={metrics['mae']:.4f}°, RMSE={metrics['rmse']:.4f}°")
        
        return self.train_summary
    
    def predict(self, X, return_uncertainty=True):
        """
        Make predictions with the trained model.
        
        Args:
            X (np.ndarray): Input features
            return_uncertainty (bool): Whether to return uncertainty estimates
            
        Returns:
            tuple or array: Predictions and optionally uncertainty estimates
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Standardize features
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_scaled[:, i] = self.scalers[i].transform(X[:, i].reshape(-1, 1)).flatten()
        
        # Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Perform predictions with uncertainty
        self.model.eval()
        predictive = Predictive(self.model, guide=self.guide, num_samples=1000)
        samples = predictive(X_tensor)
        
        # Extract predictions
        y_pred_samples = samples["obs"].cpu().numpy()
        y_pred_mean = y_pred_samples.mean(axis=0)
        
        if return_uncertainty:
            y_pred_std = y_pred_samples.std(axis=0)
            return y_pred_mean, y_pred_std
        else:
            return y_pred_mean
    
    def visualize_results(self, output_dir, experiment_name):
        """
        Create comprehensive visualizations of model results.
        
        Args:
            output_dir (str): Directory to save visualizations
            experiment_name (str): Name for this experiment
        """
        if self.train_summary is None:
            raise ValueError("Model must be trained before visualizing results")
        
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract data from training summary
        y_test = self.train_summary['y_test']
        y_pred_mean = self.train_summary['y_pred_mean']
        y_pred_std = self.train_summary['y_pred_std']
        y_pred_samples = self.train_summary['y_pred_samples']
        prior_test = self.train_summary['prior_test']
        losses = self.train_summary['losses']
        
        # Setup plots with LaTeX formatting
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        
        # Figure 1: Predicted vs True angles with uncertainty
        plt.figure(figsize=(10, 8))
        
        # Plot 1:1 line
        min_val = min(y_test.min(), y_pred_mean.min())
        max_val = max(y_test.max(), y_pred_mean.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Plot predictions with error bars
        plt.errorbar(y_test, y_pred_mean, yerr=2*y_pred_std, fmt='o', alpha=0.6, 
                     label='Bayesian Predictions (2$\sigma$ intervals)')
        
        # Plot prior method predictions
        prior_colors = {'ds': 'g', 'weighted': 'm', 'music': 'c', 'phase': 'y'}
        prior_markers = {'ds': '^', 'weighted': 's', 'music': 'd', 'phase': 'x'}
        prior_labels = {'ds': 'DS Beamforming', 'weighted': 'Weighted DS', 
                        'music': 'MUSIC', 'phase': 'Phase Difference'}
        
        for key in prior_test:
            if self.prior_type == key:
                # Highlight the method used as prior
                plt.scatter(y_test, prior_test[key], marker=prior_markers[key], 
                          color=prior_colors[key], alpha=0.5, s=80,
                          label=f"{prior_labels[key]} (Prior)")
            else:
                plt.scatter(y_test, prior_test[key], marker=prior_markers[key], 
                          color=prior_colors[key], alpha=0.3, s=40,
                          label=f"{prior_labels[key]}")
        
        plt.xlabel('True Angle (degrees)')
        plt.ylabel('Predicted Angle (degrees)')
        plt.title(f'Bayesian AoA Model with {self.prior_type.upper()} Prior\n'
                 f'MAE: {self.train_summary["mae"]:.2f}°, RMSE: {self.train_summary["rmse"]:.2f}°')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "predicted_vs_true.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Training loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('ELBO Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Prediction error distribution
        plt.figure(figsize=(12, 8))
        
        # Create subplots for each method
        plt.subplot(2, 1, 1)
        errors = y_pred_mean - y_test
        plt.hist(errors, bins=20, alpha=0.7, label='Bayesian Model')
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Prediction Error (degrees)')
        plt.ylabel('Count')
        plt.title('Error Distribution Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        for key in prior_test:
            prior_errors = prior_test[key] - y_test
            plt.hist(prior_errors, bins=20, alpha=0.5, label=prior_labels[key])
        
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Prediction Error (degrees)')
        plt.ylabel('Count')
        plt.title('Physics-based Methods Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "error_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 4: Parameter posterior distributions
        # Extract guide parameters
        posterior_means = {}
        posterior_stds = {}
        
        for name, param in self.guide.named_parameters():
            if 'AutoNormal.loc' in name and 'linear.weight' in name:
                posterior_means['weights'] = param.detach().cpu().numpy()
            elif 'AutoNormal.scale' in name and 'linear.weight' in name:
                posterior_stds['weights'] = param.detach().cpu().numpy()
            elif 'AutoNormal.loc' in name and 'linear.bias' in name:
                posterior_means['bias'] = param.detach().cpu().numpy()
            elif 'AutoNormal.scale' in name and 'linear.bias' in name:
                posterior_stds['bias'] = param.detach().cpu().numpy()
        
        # Plot weight distributions
        if 'weights' in posterior_means:
            # First, get most important features
            importance = np.abs(posterior_means['weights'][0])
            top_indices = np.argsort(importance)[-8:]  # Top 8 features
            
            plt.figure(figsize=(12, 10))
            
            for i, idx in enumerate(top_indices):
                if idx < len(self.feature_names):
                    feat_name = self.feature_names[idx]
                    mean = posterior_means['weights'][0, idx]
                    std = posterior_stds['weights'][0, idx]
                    
                    # Create range of values for x-axis
                    x = np.linspace(mean - 3*std, mean + 3*std, 1000)
                    
                    # Plot normal distribution
                    plt.subplot(4, 2, i+1)
                    plt.plot(x, norm.pdf(x, mean, std))
                    plt.axvline(0, color='r', linestyle='--', alpha=0.5)
                    plt.axvline(mean, color='g', linestyle='-')
                    plt.fill_between(x, 0, norm.pdf(x, mean, std), alpha=0.3)
                    
                    plt.title(f'{feat_name} (μ={mean:.4f}, σ={std:.4f})')
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "weight_posteriors.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Figure 5: Feature importance plot
        if 'weights' in posterior_means:
            plt.figure(figsize=(12, 8))
            
            # Calculate feature importance as absolute mean weight
            importance = np.abs(posterior_means['weights'][0])
            # Normalize to sum to 100%
            importance = 100 * importance / importance.sum()
            
            # Sort by importance
            indices = np.argsort(importance)
            sorted_importance = importance[indices]
            sorted_names = [self.feature_names[i] if i < len(self.feature_names) else f'Feature {i}' 
                           for i in indices]
            
            # Create horizontal bar chart
            plt.barh(range(len(sorted_names)), sorted_importance, align='center')
            plt.yticks(range(len(sorted_names)), sorted_names)
            plt.xlabel('Relative Importance (%)')
            plt.title('Feature Importance in Bayesian AoA Model')
            plt.grid(True, alpha=0.3)
            
            # Add percentage labels
            for i, v in enumerate(sorted_importance):
                plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Figure 6: Uncertainty analysis
        plt.figure(figsize=(10, 8))
        
        # Calculate sorted indices by uncertainty
        sorted_indices = np.argsort(y_pred_std)
        
        # Plot errors vs uncertainty
        abs_errors = np.abs(y_pred_mean - y_test)
        
        plt.scatter(y_pred_std, abs_errors, alpha=0.7)
        
        # Add trendline
        z = np.polyfit(y_pred_std, abs_errors, 1)
        p = np.poly1d(z)
        plt.plot(np.sort(y_pred_std), p(np.sort(y_pred_std)), "r--", 
                alpha=0.8, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        
        plt.xlabel('Prediction Uncertainty (std)')
        plt.ylabel('Absolute Error (degrees)')
        plt.title('Relationship Between Uncertainty and Error')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "uncertainty_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary text file
        with open(os.path.join(vis_dir, "model_summary.txt"), 'w') as f:
            f.write(f"Bayesian AoA Model Summary\n")
            f.write(f"=======================\n\n")
            
            f.write(f"Model Configuration:\n")
            f.write(f"  Prior Type: {self.prior_type}\n")
            f.write(f"  Feature Mode: {self.feature_mode}\n")
            f.write(f"  Features Used: {', '.join(self.feature_names)}\n\n")
            
            f.write(f"Performance Metrics:\n")
            f.write(f"  MAE: {self.train_summary['mae']:.4f} degrees\n")
            f.write(f"  RMSE: {self.train_summary['rmse']:.4f} degrees\n\n")
            
            f.write(f"Comparison with Physics-based Methods:\n")
            for key, metrics in self.train_summary['prior_metrics'].items():
                f.write(f"  {key.upper()}: MAE={metrics['mae']:.4f}°, RMSE={metrics['rmse']:.4f}°\n")
            
            f.write("\nModel improves over prior by: ")
            if self.prior_type in self.train_summary['prior_metrics']:
                prior_mae = self.train_summary['prior_metrics'][self.prior_type]['mae']
                improvement = prior_mae - self.train_summary['mae']
                percent = (improvement / prior_mae) * 100
                f.write(f"{improvement:.4f}° ({percent:.1f}%)\n")
            else:
                f.write("N/A (no matching prior)\n")
                
            if 'weights' in posterior_means:
                f.write("\nTop 5 Important Features:\n")
                importance = np.abs(posterior_means['weights'][0])
                top_indices = np.argsort(importance)[-5:]
                
                for i, idx in enumerate(reversed(top_indices)):
                    if idx < len(self.feature_names):
                        feat_name = self.feature_names[idx]
                        mean = posterior_means['weights'][0, idx]
                        std = posterior_stds['weights'][0, idx]
                        f.write(f"  {i+1}. {feat_name}: weight={mean:.4f}±{std:.4f}\n")
    
    def render_model_and_guide(self, output_dir, experiment_name):
        """Render the model structure and guide distributions"""
        if self.model is None or self.guide is None:
            raise ValueError("Model and guide must be trained before rendering")
            
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Render model structure as text
        model_summary = str(self.model)
        with open(os.path.join(vis_dir, "model_structure.txt"), 'w') as f:
            f.write(f"Model Structure:\n{model_summary}\n\n")
            
            # Add parameter shapes
            f.write("Parameter Shapes:\n")
            for name, param in self.model.named_parameters():
                f.write(f"  {name}: {param.shape}\n")
        
        # 2. Visualize guide distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Extract weight parameters from guide
        weight_loc = None
        weight_scale = None
        bias_loc = None
        bias_scale = None
        
        for name, param in self.guide.named_parameters():
            if 'weight' in name and 'loc' in name:
                weight_loc = param.detach().cpu().numpy()
            elif 'weight' in name and 'scale' in name:
                weight_scale = param.detach().cpu().numpy()
            elif 'bias' in name and 'loc' in name:
                bias_loc = param.detach().cpu().numpy()
            elif 'bias' in name and 'scale' in name:
                bias_scale = param.detach().cpu().numpy()
        
        # Plot weight distributions
        if weight_loc is not None and weight_scale is not None:
            # Check for negative scale values
            if np.any(weight_scale[0] < 0):
                print("Warning: Negative weight scale parameters found. Taking absolute values for visualization.")
                
            # Create parameter index
            x = np.arange(weight_loc.shape[1])
            # Plot mean with error bars - using absolute value for scale
            axes[0].errorbar(x, weight_loc[0], yerr=2*np.abs(weight_scale[0]), fmt='o', capsize=5)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(self.feature_names, rotation=90)
            axes[0].set_title(r'Weight Posterior Distributions (mean ± 2$\sigma$)')
            axes[0].grid(True, alpha=0.3)
            
            # Highlight most important features
            importance = np.abs(weight_loc[0])
            for i in range(len(x)):
                if importance[i] > np.percentile(importance, 75):
                    axes[0].annotate(self.feature_names[i], 
                                (x[i], weight_loc[0][i]),
                                xytext=(0, 10), 
                                textcoords='offset points',
                                ha='center')
        
        # Plot bias distribution
        if bias_loc is not None and bias_scale is not None:
            # Check for negative scale values
            if np.any(bias_scale < 0):
                print("Warning: Negative bias scale parameters found. Taking absolute values for visualization.")
                
            # Using absolute value for bias scale
            axes[1].errorbar([0], bias_loc, yerr=2*np.abs(bias_scale), fmt='o', capsize=5)
            axes[1].set_xticks([0])
            axes[1].set_xticklabels(['Bias'])
            axes[1].set_title(r'Bias Posterior Distribution (mean ± 2$\sigma$)')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(vis_dir, "guide_distributions.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 3. Create graphical model representation
        plt.figure(figsize=(10, 8))
        plt.title("Bayesian Linear Regression Graphical Model")
        
        # Define node positions
        pos = {
            'weights': (0.3, 0.7),
            'bias': (0.3, 0.5),
            'sigma': (0.3, 0.3),
            'mean': (0.6, 0.5),
            'y': (0.9, 0.5)
        }
        
        # Create nodes
        node_labels = {
            'weights': 'Weights\nw ~ N(prior_mean, prior_std)',
            'bias': 'Bias\nb ~ N(0, 1)',
            'sigma': 'Noise\nsigma ~ LogNormal(0, 1)',
            'mean': 'Linear Function\n mu = X·w + b',
            'y': 'Observations\ny ~ N(mu, sigma)'
        }
        
        # Create edges
        edges = [
            ('weights', 'mean'),
            ('bias', 'mean'),
            ('mean', 'y'),
            ('sigma', 'y')
        ]
        
        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(pos.keys())
        G.add_edges_from(edges)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
        
        # Add labels with custom positioning
        label_pos = {k: (v[0], v[1]-0.02) for k, v in pos.items()}
        nx.draw_networkx_labels(G, label_pos, labels=node_labels, font_size=10, 
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "graphical_model.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model and guide renderings saved to {vis_dir}")

    def visualize_prior_vs_posterior(self, output_dir, experiment_name, num_samples=5):
        """
        Visualize how the model transforms prior distributions into posterior distributions.
        
        Args:
            output_dir (str): Directory to save visualizations
            experiment_name (str): Name for this experiment
            num_samples (int): Number of test samples to visualize
        """
        if self.train_summary is None:
            raise ValueError("Model must be trained before visualizing")
            
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract data from training summary
        y_test = self.train_summary['y_test']
        y_pred_mean = self.train_summary['y_pred_mean']
        y_pred_std = self.train_summary['y_pred_std']
        prior_test = self.train_summary['prior_test']
        
        # Get the prior standard deviation used during training
        # Handle the case when the prior type is 'flat' or not in prior_metrics
        if self.prior_type in self.train_summary['prior_metrics']:
            prior_error = self.train_summary['prior_metrics'][self.prior_type]['mae']
            prior_std = max(0.1, prior_error)
        else:
            # For 'flat' prior or any other prior not in prior_metrics, use a default value
            prior_std = 1.0  # Default value
            print(f"Using default prior std={prior_std} for {self.prior_type} prior")
        
        # Select sample indices - try to get a diverse set
        if len(y_test) <= num_samples:
            indices = np.arange(len(y_test))
        else:
            # Sort by true angle and select evenly spaced samples
            sorted_indices = np.argsort(y_test)
            step = len(sorted_indices) // num_samples
            indices = sorted_indices[::step][:num_samples]
        
        # Create figure
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        # Plot each sample
        for i, idx in enumerate(indices):
            ax = axes[i]
            
            # Get values for this sample
            true_angle = y_test[idx]
            posterior_mean = y_pred_mean[idx]
            posterior_std = y_pred_std[idx]
            
            # Get prior value based on prior type
            if self.prior_type in prior_test:
                prior_mean = prior_test[self.prior_type][idx]
            else:
                # For 'flat' prior or any other prior not in prior_test
                prior_mean = 0.0  # Use 0 as default for flat prior
            
            # Create x-axis range for plotting distributions
            plot_range = 4 * max(prior_std, posterior_std)
            x = np.linspace(min(prior_mean, posterior_mean) - plot_range, 
                            max(prior_mean, posterior_mean) + plot_range, 1000)
            
            # Plot prior distribution
            prior_pdf = norm.pdf(x, prior_mean, prior_std)
            ax.plot(x, prior_pdf, 'r--', linewidth=2, label=f'{self.prior_type.upper()} Prior')
            ax.fill_between(x, 0, prior_pdf, color='red', alpha=0.2)
            
            # Plot posterior distribution
            posterior_pdf = norm.pdf(x, posterior_mean, posterior_std)
            ax.plot(x, posterior_pdf, 'b-', linewidth=2, label='Bayesian Posterior')
            ax.fill_between(x, 0, posterior_pdf, color='blue', alpha=0.2)
            
            # Plot true angle
            ax.axvline(true_angle, color='k', linestyle='-', linewidth=2, label='True Angle')
            
            # Add labels and legend
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'Sample {i+1}: Prior vs Posterior (True Angle: {true_angle:.2f}°)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Annotate statistics
            prior_error = abs(prior_mean - true_angle)
            posterior_error = abs(posterior_mean - true_angle)
            
            # Avoid division by zero
            if prior_error > 0:
                improvement = prior_error - posterior_error
                improvement_pct = 100*improvement/prior_error
            else:
                improvement = posterior_error
                improvement_pct = 0
            
            stats_text = (f'Prior: $\\mu={prior_mean:.2f}^\\circ$, $\\sigma={prior_std:.2f}^\\circ$, Error$={prior_error:.2f}^\\circ$\n'
                        f'Posterior: $\\mu={posterior_mean:.2f}^\\circ$, $\\sigma={posterior_std:.2f}^\\circ$, Error$={posterior_error:.2f}^\\circ$\n'
                        f'Improvement: ${improvement:.2f}^\\circ$ ({improvement_pct:.1f}\\% reduction)')
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save figure
        plt.tight_layout()
        fig.savefig(os.path.join(vis_dir, "prior_vs_posterior.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Prior vs posterior visualization saved to {vis_dir}")

    def plot_posterior_predictive(self, output_dir, experiment_name):
        """Plot posterior predictive distribution"""
        if self.train_summary is None:
            raise ValueError("Model must be trained before plotting posterior predictive")
            
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract data
        X_test = self.train_summary['X_test']
        y_test = self.train_summary['y_test']
        y_pred_samples = self.train_summary['y_pred_samples']
        
        # Plot posterior predictive distribution
        plt.figure(figsize=(12, 8))
        
        # Sort test points by true angle for better visualization
        sort_idx = np.argsort(y_test)
        y_test_sorted = y_test[sort_idx]
        
        # Calculate percentiles of predictions
        y_pred_5 = np.percentile(y_pred_samples[:, sort_idx], 5, axis=0)
        y_pred_95 = np.percentile(y_pred_samples[:, sort_idx], 95, axis=0)
        y_pred_25 = np.percentile(y_pred_samples[:, sort_idx], 25, axis=0)
        y_pred_75 = np.percentile(y_pred_samples[:, sort_idx], 75, axis=0)
        y_pred_50 = np.percentile(y_pred_samples[:, sort_idx], 50, axis=0)
        
        # Plot the data
        plt.fill_between(range(len(y_test)), y_pred_5, y_pred_95, alpha=0.3, color='blue', 
                        label='90% Credible Interval')
        plt.fill_between(range(len(y_test)), y_pred_25, y_pred_75, alpha=0.5, color='blue', 
                        label='50% Credible Interval')
        plt.plot(range(len(y_test)), y_pred_50, 'b-', linewidth=2, label='Median Prediction')
        plt.plot(range(len(y_test)), y_test_sorted, 'ro', label='True Angles')
        
        plt.xlabel('Test Point Index (sorted by true angle)')
        plt.ylabel('Angle (degrees)')
        plt.title('Posterior Predictive Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "posterior_predictive.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_uncertainty_calibration(self, output_dir, experiment_name):
        """Create a calibration plot for uncertainty estimates"""
        if self.train_summary is None:
            raise ValueError("Model must be trained before plotting uncertainty calibration")
            
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract data
        y_test = self.train_summary['y_test']
        y_pred_mean = self.train_summary['y_pred_mean']
        y_pred_std = self.train_summary['y_pred_std']
        
        # Calculate standardized errors
        z_scores = (y_test - y_pred_mean) / y_pred_std
        
        # Create calibration plot
        plt.figure(figsize=(10, 8))
        
        # Plot histogram of standardized errors
        plt.hist(z_scores, bins=20, density=True, alpha=0.6, label='Standardized Errors')
        
        # Plot standard normal PDF for comparison
        x = np.linspace(-4, 4, 1000)
        plt.plot(x, norm.pdf(x), 'r-', linewidth=2, label='Standard Normal')
        
        plt.xlabel('Standardized Error (z-score)')
        plt.ylabel('Density')
        plt.title('Uncertainty Calibration - Standard Normal Check')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add KS test result
        ks_stat, ks_pval = kstest(z_scores, 'norm')
        plt.text(0.05, 0.95, f'KS Test: stat={ks_stat:.3f}, p-value={ks_pval:.3f}',
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "uncertainty_calibration.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_weight_distributions(self, output_dir, experiment_name):
        """
        Visualize the prior and posterior distributions of model weights,
        focusing on parameters related to beamforming.
        
        Args:
            output_dir (str): Directory to save visualizations
            experiment_name (str): Name for this experiment
        """
        if self.model is None or self.guide is None:
            raise ValueError("Model must be trained before visualizing weights")
            
        # Create output directory
        vis_dir = os.path.join(output_dir, "bayesian_model", experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract prior parameters (used during model initialization)
        # The prior standard deviation used for weights in _build_model
        if self.prior_type in self.train_summary['prior_metrics']:
            prior_error = self.train_summary['prior_metrics'][self.prior_type]['mae']
            prior_std = max(0.1, prior_error)
        else:
            prior_std = 1.0  # Default value for 'flat' prior
        prior_mean = 0.0  # We typically use zero-centered priors
        
        # Extract posterior parameters from guide
        weight_loc = None
        weight_scale = None
        
        for name, param in self.guide.named_parameters():
            if 'weight' in name and 'loc' in name:
                weight_loc = param.detach().cpu().numpy()
            elif 'weight' in name and 'scale' in name:
                weight_scale = param.detach().cpu().numpy()
        
        if weight_loc is None or weight_scale is None:
            print("Could not extract posterior weight parameters")
            return
        
        # Identify beamforming-related features (phase differences, RSSI)
        beamforming_indices = []
        phase_indices = []
        rssi_indices = []
        
        for i, name in enumerate(self.feature_names):
            if 'phase' in name.lower():
                phase_indices.append(i)
                beamforming_indices.append(i)
            elif 'rssi' in name.lower():
                rssi_indices.append(i)
                beamforming_indices.append(i)
        
        # Create visualization for beamforming-related weights
        plt.figure(figsize=(14, 10))
        
        # Determine number of subplots needed
        n_plots = len(beamforming_indices)
        if n_plots == 0:
            print("No beamforming-related features found")
            return
        
        # Determine grid layout
        rows = int(np.ceil(np.sqrt(n_plots)))
        cols = int(np.ceil(n_plots / rows))
        
        for i, idx in enumerate(beamforming_indices):
            # Create subplot
            plt.subplot(rows, cols, i+1)
            
            # Get feature name and parameters
            feature_name = self.feature_names[idx]
            post_mean = weight_loc[0, idx]
            post_std = weight_scale[0, idx]
            
            # Create x-axis range that accounts for both distributions
            # Create separate ranges for each distribution
            prior_range = np.linspace(prior_mean - 4*prior_std, prior_mean + 4*prior_std, 1000)
            post_range = np.linspace(post_mean - 4*post_std, post_mean + 4*post_std, 1000)
            
            # Plot distributions on their own appropriate scales
            ax1 = plt.gca()
            ax2 = ax1.twinx()  # Create second y-axis
            
            # Plot prior on first axis
            prior_pdf = norm.pdf(prior_range, prior_mean, prior_std)
            ax1.plot(prior_range, prior_pdf, 'r-', linewidth=2, label='Prior')
            ax1.fill_between(prior_range, 0, prior_pdf, color='red', alpha=0.2)
            ax1.set_ylabel('Prior Density', color='r')
            ax1.tick_params(axis='y', labelcolor='r')
            
            # Plot posterior on second axis with different scale
            post_pdf = norm.pdf(post_range, post_mean, post_std)
            ax2.plot(post_range, post_pdf, 'b-', linewidth=2, label='Posterior')
            ax2.fill_between(post_range, 0, post_pdf, color='blue', alpha=0.2)
            ax2.set_ylabel('Posterior Density', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            
            # Add vertical lines for means
            plt.axvline(prior_mean, color='r', linestyle='--', alpha=0.7)
            plt.axvline(post_mean, color='b', linestyle='--', alpha=0.7)
            
            # Add annotations
            plt.title(f'{feature_name}')
            plt.figtext(0.02, 0.98, f'Prior: $\\mu$={prior_mean:.3f}, $\\sigma$={prior_std:.3f}\n'
                                f'Post: $\\mu$={post_mean:.3f}, $\\sigma$={post_std:.3f}',
                    horizontalalignment='left', verticalalignment='top')
            
            # Enhanced annotation for clearer comparison
            """
            plt.annotate(f'Prior: $\\mu$={prior_mean:.2f}, $\\sigma$={prior_std:.2f}\n'
                        f'Posterior: $\\mu$={post_mean:.2f}, $\\sigma$={post_std:.2f}\n'
                        f'Diff: {post_mean-prior_mean:.2f}',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        ha='left', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            """
        
        # Add overall title
        plt.suptitle(f'Prior vs Posterior Weight Distributions for Beamforming-Related Features\n'
                    f'Model: {self.prior_type.upper()} Prior with {self.feature_mode} features',
                    fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(os.path.join(vis_dir, "beamforming_weight_distributions.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create separate plots for phase-related and RSSI-related features
        if phase_indices:
            self._plot_feature_group(phase_indices, "Phase-Related", prior_mean, prior_std, 
                                    weight_loc, weight_scale, vis_dir)
        
        if rssi_indices:
            self._plot_feature_group(rssi_indices, "RSSI-Related", prior_mean, prior_std, 
                                    weight_loc, weight_scale, vis_dir)
        
        print(f"Weight distribution visualizations saved to {vis_dir}")

    def _plot_feature_group(self, indices, group_name, prior_mean, prior_std, weight_loc, weight_scale, vis_dir):
        """Helper method to plot a group of related feature weight distributions"""
        plt.figure(figsize=(12, 8))
        
        rows = int(np.ceil(np.sqrt(len(indices))))
        cols = int(np.ceil(len(indices) / rows))
        
        for i, idx in enumerate(indices):
            plt.subplot(rows, cols, i+1)
            
            feature_name = self.feature_names[idx]
            post_mean = weight_loc[0, idx]
            post_std = weight_scale[0, idx]
            
            # Create x-axis range for plotting distributions
            plot_range = 5 * max(prior_std, post_std)
            x = np.linspace(min(prior_mean, post_mean) - plot_range, 
                            max(prior_mean, post_mean) + plot_range, 1000)
            
            # Plot prior distribution
            prior_pdf = norm.pdf(x, prior_mean, prior_std)
            plt.plot(x, prior_pdf, 'r--', linewidth=1.5, label='Prior')
            plt.fill_between(x, 0, prior_pdf, color='red', alpha=0.15)
            
            # Plot posterior distribution
            posterior_pdf = norm.pdf(x, post_mean, post_std)
            plt.plot(x, posterior_pdf, 'b-', linewidth=2.0, label='Posterior', zorder=10)
            plt.fill_between(x, 0, posterior_pdf, color='blue', alpha=0.25, zorder=5)
            
            plt.axvline(prior_mean, color='r', linestyle='-', alpha=0.5, label = '_Prior Mean')
            plt.axvline(post_mean, color='b', linestyle='-', alpha=0.7, label = '_Posterior Mean')
            plt.axvline(0, color='k', linestyle=':', alpha=0.5, label = '_Zero')
            
            plt.xlabel('Weight Value')
            plt.ylabel('Probability Density')
            plt.title(f'{feature_name}')
            if i == 0:
                plt.legend()
        
        plt.suptitle(f'{group_name} Feature Weight Distributions\n'
                    f'Model: {self.prior_type.upper()} Prior with {self.feature_mode} features',
                    fontsize=14)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(vis_dir, f"{group_name.lower().replace('-','_')}_weights.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_with_posterior_weights(self, data_manager, output_dir, experiment_name):
        """
        Re-analyze antenna array data using the posterior weights from the Bayesian model.
        
        Args:
            data_manager: DataManager object with the dataset
            output_dir: Directory to save results
            experiment_name: Name for this experiment
        """
        if self.model is None:
            raise ValueError("Model must be trained before analysis")
        
        # Create output directory
        analysis_dir = os.path.join(output_dir, "bayesian_model", experiment_name, "posterior_weighted_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Extract posterior weights
        weight_loc = None
        for name, param in self.guide.named_parameters():
            if 'weight' in name and 'loc' in name:
                weight_loc = param.detach().cpu().numpy()
        
        if weight_loc is None:
            raise ValueError("Could not extract posterior weights")
        
        # Select a few test examples
        if data_manager.results is None:
            data_manager.analyze_all_data(save_results=False)
        
        # Get features for a few test cases
        test_indices = np.linspace(0, len(data_manager.metadata)-1, 5, dtype=int)
        
        for idx in test_indices:
            # Get original analysis results
            meta = data_manager.metadata.iloc[idx]
            signals = data_manager.signal_data[idx]
            
            # Extract parameters
            D = meta['D']
            W = meta['W']
            L = meta['L']
            wavelength = meta['lambda']
            true_angle = data_manager.get_true_angle(D, W)
            
            # Extract signals
            phasor1 = signals['phasor1']
            phasor2 = signals['phasor2']
            rssi1 = signals['rssi1']
            rssi2 = signals['rssi2']
            
            # Run original analysis with FULL range
            analysis_aoa = np.arange(main.MIN_ANGLE, main.MAX_ANGLE + main.STEP, main.STEP)
            original_results = main.analyze_aoa(
                phasor1, phasor2, rssi1, rssi2, 
                L, wavelength, analysis_aoa, true_angle
            )
            
            # Create feature vector for this example using the same feature extraction logic
            features = self._extract_features_for_sample(
                phasor1, phasor2, rssi1, rssi2, D, W, wavelength
            )
            
            # Scale features
            scaled_features = np.zeros_like(features)
            for i in range(features.shape[0]):
                scaled_features[i] = self.scalers[i].transform([[features[i]]])[0][0]
            
            # Get model prediction
            X_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prediction = self.model(X_tensor).item()
            
            # Use RESTRICTED range for Bayesian analysis
            bayesian_aoa = np.arange(main.BAYESIAN_MIN_ANGLE, main.BAYESIAN_MAX_ANGLE + main.STEP, main.STEP)
            batch_size = len(bayesian_aoa)
            feature_batch = np.tile(scaled_features, (batch_size, 1))
            
            # Modify angle-specific features for each item in the batch
            for i, angle in enumerate(bayesian_aoa):
                # Only modify features that relate to angle
                feature_batch[i, 0] = np.sin(np.deg2rad(angle))  # Assuming first feature is sin(phi)
                feature_batch[i, 1] = np.cos(np.deg2rad(angle))  # Assuming second feature is cos(phi)
            
            # Convert to tensor and get predictions in one batch
            batch_tensor = torch.tensor(feature_batch, dtype=torch.float32).to(self.device)
            
            # Process in smaller chunks to avoid memory issues
            chunk_size = 100
            weighted_spectrum = np.zeros(batch_size)
            
            print("Processing angles in batches...")
            for start_idx in range(0, batch_size, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size)
                chunk = batch_tensor[start_idx:end_idx]
                
                with torch.no_grad():
                    # Use a simpler forward pass that doesn't sample
                    chunk_output = self.model.linear(chunk).squeeze(-1)
                    weighted_spectrum[start_idx:end_idx] = chunk_output.cpu().numpy()
            
            # Normalize weighted spectrum
            weighted_spectrum = np.exp(-(weighted_spectrum - prediction)**2)
            weighted_spectrum = weighted_spectrum / weighted_spectrum.max()
            
            # Find peak of weighted spectrum
            weighted_angle = bayesian_aoa[np.argmax(weighted_spectrum)]
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Spectrum comparison
            plt.subplot(2, 1, 1)
            plt.plot(analysis_aoa, original_results['spectra']['ds'], 'r-', label='Original DS')
            plt.plot(analysis_aoa, original_results['spectra']['weighted'], 'm--', label='Original Weighted')
            
            # Create a padded version of the Bayesian spectrum to match the full range
            full_weighted_spectrum = np.zeros(len(analysis_aoa))
            # Find the indices that match the Bayesian range
            idx_start = np.searchsorted(analysis_aoa, main.BAYESIAN_MIN_ANGLE)
            idx_end = np.searchsorted(analysis_aoa, main.BAYESIAN_MAX_ANGLE)
            # Only fill in the values within the Bayesian range
            step_ratio = len(bayesian_aoa) / (idx_end - idx_start)
            for i in range(idx_start, idx_end):
                bayesian_idx = int((i - idx_start) * step_ratio)
                if bayesian_idx < len(weighted_spectrum):
                    full_weighted_spectrum[i] = weighted_spectrum[bayesian_idx]
            
            # Plot the limited Bayesian spectrum
            plt.plot(analysis_aoa, full_weighted_spectrum, 'g-', linewidth=2, label='Bayesian Weighted (±15°)')
            
            # Add shaded region to show Bayesian analysis range
            plt.axvspan(main.BAYESIAN_MIN_ANGLE, main.BAYESIAN_MAX_ANGLE, color='lightgreen', alpha=0.2, label='Bayesian Range')
            
            # Add vertical lines for angle estimates
            plt.axvline(original_results['angles']['ds'], color='r', linestyle=':', label='DS Est.')
            plt.axvline(original_results['angles']['weighted'], color='m', linestyle=':', label='Weighted Est.')
            plt.axvline(weighted_angle, color='g', linestyle=':', label='Bayesian Est.')
            plt.axvline(true_angle, color='k', linestyle='-', label='True Angle')
            
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Normalized Power')
            plt.title(f'Spectrum Comparison (D={D:.2f}m, W={W:.2f}m)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 2: Error comparison
            plt.subplot(2, 1, 2)
            methods = ['Phase', 'DS', 'Weighted', 'MUSIC', 'Bayesian']
            errors = [
                abs(original_results['angles']['phase'] - true_angle),
                abs(original_results['angles']['ds'] - true_angle),
                abs(original_results['angles']['weighted'] - true_angle),
                abs(original_results['angles']['music'] - true_angle),
                abs(weighted_angle - true_angle)
            ]
            
            plt.bar(methods, errors)
            plt.ylabel('Absolute Error (degrees)')
            plt.title('Error Comparison')
            plt.grid(True, alpha=0.3)
            
            # Add error values as text
            for i, v in enumerate(errors):
                plt.text(i, v + 0.1, f'{v:.2f}°', ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f'posterior_weighted_analysis_D{D:.2f}_W{W:.2f}.png'), 
                    dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Completed analysis for sample at D={D:.2f}m, W={W:.2f}m")
        
        print(f"Posterior weighted analysis completed and saved to {analysis_dir}")

        # Inside BayesianAoARegressor class:
    def _extract_features_for_sample(self, phasor1, phasor2, rssi1, rssi2, D, W, wavelength):
        """Extract features for a single sample"""
        # Mean phase and magnitude values
        phase1_mean = np.angle(np.mean(phasor1))
        phase2_mean = np.angle(np.mean(phasor2))
        mag1_mean = np.mean(np.abs(phasor1))
        mag2_mean = np.mean(np.abs(phasor2))
        
        # Phase difference
        phase_diff = phase1_mean - phase2_mean
        phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap to [-π, π]
        
        # RSSI features
        rssi1_mean = np.mean(rssi1)
        rssi2_mean = np.mean(rssi2)
        rssi_diff = rssi1_mean - rssi2_mean
        
        # Phasor correlations
        phasor1_real_mean = np.mean(phasor1.real)
        phasor1_imag_mean = np.mean(phasor1.imag)
        phasor2_real_mean = np.mean(phasor2.real)
        phasor2_imag_mean = np.mean(phasor2.imag)
        
        # Create feature vector
        features = np.array([
            phase1_mean, phase2_mean, phase_diff,
            mag1_mean, mag2_mean,
            rssi1_mean, rssi2_mean, rssi_diff,
            phasor1_real_mean, phasor1_imag_mean,
            phasor2_real_mean, phasor2_imag_mean,
            wavelength
        ])
        
        # Add geometric features based on feature_mode
        if self.feature_mode in ['full', 'width_only']:
            features = np.append(features, W)
        
        if self.feature_mode == 'full':
            features = np.append(features, D)
        
        return features

def train_bayesian_models(data_manager, results_dir, num_epochs=10000):
    """
    Train multiple Bayesian AoA regression models with different priors and feature sets.
    
    Args:
        data_manager: DataManager object containing the dataset
        results_dir: Directory to save results
        num_epochs: Number of training epochs
        
    Returns:
        dict: Dictionary containing trained models and results
    """
    print("\n=== TRAINING BAYESIAN AOA REGRESSION MODELS ===")
    
    # Define configurations to test - correctly including all feature modes
    configs = [
        # Full features (includes both distance and width)
        {"prior": "ds", "features": "full", "name": "ds_full"},
        {"prior": "music", "features": "full", "name": "music_full"},
        {"prior": "weighted", "features": "full", "name": "weighted_full"},
        {"prior": "flat", "features": "full", "name": "flat_full"},
        
        # Width only (includes width but not distance)
        {"prior": "ds", "features": "width_only", "name": "ds_width"},
        {"prior": "music", "features": "width_only", "name": "music_width"},
        {"prior": "weighted", "features": "width_only", "name": "weighted_width"},
        {"prior": "flat", "features": "width_only", "name": "flat_width"},
        
        # Sensor only (no distance or width)
        {"prior": "ds", "features": "sensor_only", "name": "ds_sensor"},
        {"prior": "music", "features": "sensor_only", "name": "music_sensor"},
        {"prior": "weighted", "features": "sensor_only", "name": "weighted_sensor"},
        {"prior": "flat", "features": "sensor_only", "name": "flat_sensor"}
    ]
    
    # Dictionary to store results
    models = {}
    results = {}
    
    # Train models for each configuration
    for config in configs:
        print(f"\n--- Training Bayesian model with {config['prior']} prior, {config['features']} features ---")
        
        # Create model
        model = BayesianAoARegressor(
            use_gpu=True,
            prior_type=config['prior'],
            feature_mode=config['features']
        )
        
        # Train model
        train_results = model.train(data_manager, num_epochs=num_epochs)

        # Store model and results
        models[config['name']] = model
        results[config['name']] = train_results
        
        # Visualize results
        model.visualize_results(results_dir, config['name'])

        # Generate new visualizations
        model.render_model_and_guide(results_dir, config['name'])
        model.plot_posterior_predictive(results_dir, config['name'])
        model.plot_uncertainty_calibration(results_dir, config['name'])

        # Visualize prior vs posterior
        models[config['name']].visualize_prior_vs_posterior(results_dir, config['name'])
        model.visualize_weight_distributions(results_dir, config['name'])
        model.analyze_with_posterior_weights(data_manager, results_dir, config['name'])
        
        print(f"Completed training {config['name']}")
    
    # Create comparison visualizations
    compare_bayesian_models(models, results, results_dir)
    
    return {"models": models, "results": results}

def compare_bayesian_models(models, results, output_dir):
    """
    Create comparative visualizations for multiple Bayesian models.
    
    Args:
        models (dict): Dictionary of trained BayesianAoARegressor models
        results (dict): Dictionary of training results
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    comp_dir = os.path.join(output_dir, "bayesian_model_comparison")
    os.makedirs(comp_dir, exist_ok=True)
    
    # Setup plots with LaTeX formatting
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # Extract performance metrics
    model_names = []
    maes = []
    rmses = []
    prior_maes = {}
    
    # Group by prior type and feature mode
    prior_types = set()
    feature_modes = set()
    grouped_results = {}
    
    for name, res in results.items():
        model = models[name]
        
        # Track model info
        model_names.append(name)
        maes.append(res['mae'])
        rmses.append(res['rmse'])
        
        # Extract prior metrics
        if model.prior_type in res['prior_metrics']:
            if model.prior_type not in prior_maes:
                prior_maes[model.prior_type] = []
            prior_maes[model.prior_type].append(res['prior_metrics'][model.prior_type]['mae'])
        
        # Group by prior and features
        prior_types.add(model.prior_type)
        feature_modes.add(model.feature_mode)
        
        key = (model.prior_type, model.feature_mode)
        grouped_results[key] = {
            'name': name,
            'mae': res['mae'],
            'rmse': res['rmse'],
            'y_test': res['y_test'],
            'y_pred': res['y_pred_mean'],
            'y_std': res['y_pred_std']
        }
    
    # Figure 1: Overall Performance Comparison
    plt.figure(figsize=(12, 8))
    
    # Sort models by MAE
    sort_idx = np.argsort(maes)
    sorted_names = [model_names[i] for i in sort_idx]
    sorted_maes = [maes[i] for i in sort_idx]
    sorted_rmses = [rmses[i] for i in sort_idx]
    
    # Create readable labels
    display_names = []
    for name in sorted_names:
        parts = name.split('_')
        prior = parts[0].upper()
        if parts[1] == 'full':
            features = 'Full'
        elif parts[1] == 'width':
            features = 'Width Only'
        else:
            features = 'Sensor Only'
        display_names.append(f"{prior} + {features}")
    
    # Create bar chart
    x = np.arange(len(display_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, sorted_maes, width, label='MAE')
    rects2 = ax.bar(x + width/2, sorted_rmses, width, label='RMSE')
    
    # Add labels and title
    ax.set_ylabel('Error (degrees)')
    ax.set_title('Bayesian AoA Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}°',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(comp_dir, "overall_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Prior vs Bayesian Improvement
    if prior_maes:
        plt.figure(figsize=(10, 6))
        
        # For each prior type, show improvement
        x_labels = []
        improvements = []
        
        for prior_type in prior_maes:
            for i, (name, res) in enumerate(results.items()):
                model = models[name]
                if model.prior_type == prior_type:
                    # Calculate improvement
                    prior_mae = res['prior_metrics'][prior_type]['mae']
                    model_mae = res['mae']
                    improvement = prior_mae - model_mae
                    percent = (improvement / prior_mae) * 100 if prior_mae > 0 else 0
                    
                    # Create label
                    if model.feature_mode == 'full':
                        label = f"{prior_type.upper()} + Full"
                    elif model.feature_mode == 'width_only':
                        label = f"{prior_type.upper()} + Width"
                    else:
                        label = f"{prior_type.upper()} + Sensor"
                    
                    x_labels.append(label)
                    improvements.append(percent)
        
        # Sort by improvement
        sort_idx = np.argsort(improvements)
        sorted_labels = [x_labels[i] for i in sort_idx]
        sorted_improvements = [improvements[i] for i in sort_idx]
        
        # Plot improvements
        plt.barh(range(len(sorted_labels)), sorted_improvements, align='center')
        plt.yticks(range(len(sorted_labels)), sorted_labels)
        plt.xlabel('Improvement Over Prior (%)')
        plt.title('Bayesian Model Improvement Over Physics-Based Priors')
        plt.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, v in enumerate(sorted_improvements):
            plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, "prior_improvement.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 3: Scatter plot matrix for feature mode comparison
    for prior in prior_types:
        # Skip flat prior
        if prior == 'flat':
            continue
            
        # Find all models with this prior but different feature modes
        models_with_prior = [(key, data) for key, data in grouped_results.items() 
                            if key[0] == prior]
        
        if len(models_with_prior) > 1:
            fig, axes = plt.subplots(1, len(models_with_prior), figsize=(15, 5))
            if len(models_with_prior) == 1:
                axes = [axes]
                
            fig.suptitle(f'{prior.upper()} Prior with Different Feature Sets', fontsize=16)
            
            for i, ((_, feat_mode), data) in enumerate(models_with_prior):
                # Create scatter plot
                ax = axes[i]
                ax.scatter(data['y_test'], data['y_pred'], alpha=0.7)
                
                # Add 1:1 line
                min_val = min(data['y_test'].min(), data['y_pred'].min())
                max_val = max(data['y_test'].max(), data['y_pred'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                # Add labels
                if feat_mode == 'full':
                    title = 'Full Features'
                elif feat_mode == 'width_only':
                    title = 'Width Only'
                else:
                    title = 'Sensor Only'
                    
                ax.set_title(f'{title} (MAE: {data["mae"]:.2f}°)')
                ax.set_xlabel('True Angle (degrees)')
                ax.set_ylabel('Predicted Angle (degrees)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(comp_dir, f"{prior}_feature_comparison.png"), 
                      dpi=300, bbox_inches='tight')
            plt.close()
    
    # Figure 4: Feature mode comparison across priors
    for feat_mode in feature_modes:
        # Find all models with this feature mode but different priors
        models_with_feat = [(key, data) for key, data in grouped_results.items() 
                           if key[1] == feat_mode]
        
        if len(models_with_feat) > 1:
            fig, axes = plt.subplots(1, len(models_with_feat), figsize=(15, 5))
            if len(models_with_feat) == 1:
                axes = [axes]
                
            feat_title = "Full Features" if feat_mode == "full" else \
                         "Width Only" if feat_mode == "width_only" else "Sensor Only"
            
            fig.suptitle(f'{feat_title} with Different Priors', fontsize=16)
            
            for i, ((prior, _), data) in enumerate(models_with_feat):
                # Create scatter plot
                ax = axes[i]
                ax.scatter(data['y_test'], data['y_pred'], alpha=0.7)
                
                # Add 1:1 line
                min_val = min(data['y_test'].min(), data['y_pred'].min())
                max_val = max(data['y_test'].max(), data['y_pred'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                # Add labels
                title = f'{prior.upper()} Prior'
                    
                ax.set_title(f'{title} (MAE: {data["mae"]:.2f}°)')
                ax.set_xlabel('True Angle (degrees)')
                ax.set_ylabel('Predicted Angle (degrees)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(comp_dir, f"{feat_mode}_prior_comparison.png"), 
                      dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create summary file
    with open(os.path.join(comp_dir, "comparison_summary.txt"), 'w') as f:
        f.write("Bayesian AoA Model Comparison\n")
        f.write("===========================\n\n")
        
        f.write("Performance Summary:\n")
        for i, name in enumerate(sorted_names):
            model = models[name]
            mae = sorted_maes[i]
            rmse = sorted_rmses[i]
            
            # Format name for readability
            parts = name.split('_')
            prior = parts[0].upper()
            if parts[1] == 'full':
                features = 'Full Features'
            elif parts[1] == 'width':
                features = 'Width Only'
            else:
                features = 'Sensor Only'
                
            f.write(f"{i+1}. {prior} Prior with {features}\n")
            f.write(f"   MAE: {mae:.4f}°, RMSE: {rmse:.4f}°\n")
            
            # Add improvement over prior if applicable
            if model.prior_type in results[name]['prior_metrics']:
                prior_mae = results[name]['prior_metrics'][model.prior_type]['mae']
                improvement = prior_mae - mae
                percent = (improvement / prior_mae) * 100 if prior_mae > 0 else 0
                f.write(f"   Improvement over {model.prior_type.upper()} prior: {improvement:.4f}° ({percent:.1f}%)\n")
            
            f.write("\n")
        
        # Find best overall model
        best_idx = np.argmin(maes)
        best_name = model_names[best_idx]
        best_model = models[best_name]
        
        f.write("\nBest Overall Model:\n")
        f.write(f"  {best_model.prior_type.upper()} Prior with {best_model.feature_mode} features\n")
        f.write(f"  MAE: {maes[best_idx]:.4f}°, RMSE: {rmses[best_idx]:.4f}°\n\n")
        
        # Best model by feature mode
        f.write("Best Model by Feature Set:\n")
        for feat_mode in feature_modes:
            feat_models = [(name, res['mae']) for name, res in results.items() 
                          if models[name].feature_mode == feat_mode]
            if feat_models:
                best_feat_name, best_feat_mae = min(feat_models, key=lambda x: x[1])
                best_feat_model = models[best_feat_name]
                
                if feat_mode == 'full':
                    feat_desc = 'Full Features'
                elif feat_mode == 'width_only':
                    feat_desc = 'Width Only'
                else:
                    feat_desc = 'Sensor Only'
                    
                f.write(f"  {feat_desc}: {best_feat_model.prior_type.upper()} Prior (MAE: {best_feat_mae:.4f}°)\n")
        
        f.write("\n")
        
        # Best model by prior type
        f.write("Best Model by Prior Type:\n")
        for prior in prior_types:
            prior_models = [(name, res['mae']) for name, res in results.items() 
                           if models[name].prior_type == prior]
            if prior_models:
                best_prior_name, best_prior_mae = min(prior_models, key=lambda x: x[1])
                best_prior_model = models[best_prior_name]
                
                f.write(f"  {prior.upper()} Prior: {best_prior_model.feature_mode} features (MAE: {best_prior_mae:.4f}°)\n")