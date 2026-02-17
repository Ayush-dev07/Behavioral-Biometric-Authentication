"""
Configuration management for keystroke authentication system.
Centralizes all settings for easy modification.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Neural network model configuration."""
    
    input_dim: int = 4              # Number of features per keystroke
    hidden_dim1: int = 128          # First LSTM layer size
    hidden_dim2: int = 64           # Second LSTM layer size
    embedding_dim: int = 32         # Embedding vector size
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    
    # Triplet loss
    margin: float = 0.2             # Triplet loss margin


@dataclass
class AuthConfig:
    """Authentication configuration."""
    
    # Enrollment
    enrollment_samples: int = 3     # Number of samples for enrollment
    augment_enrollment: bool = True # Augment enrollment data
    augmentation_factor: int = 2    # Augmentations per sample
    
    # Thresholds
    global_threshold: float = 0.75  # Initial threshold for all users
    user_threshold_samples: int = 50 # Samples before user-specific threshold
    min_threshold: float = 0.5      # Minimum allowed threshold
    
    # Security
    max_failed_attempts: int = 5    # Max failed attempts before lockout
    lockout_duration: int = 300     # Lockout duration in seconds


@dataclass
class ServerConfig:
    """Flask server configuration."""
    
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = True
    
    # CORS
    cors_enabled: bool = True
    cors_origins: str = '*'
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60


@dataclass
class StorageConfig:
    """Data storage configuration."""
    
    data_dir: str = 'data'
    models_dir: str = 'models'
    
    # File names
    users_file: str = 'users.pkl'
    thresholds_file: str = 'thresholds.pkl'
    model_file: str = 'siamese_model.pt'
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    @property
    def users_path(self) -> str:
        return os.path.join(self.data_dir, self.users_file)
    
    @property
    def thresholds_path(self) -> str:
        return os.path.join(self.data_dir, self.thresholds_file)
    
    @property
    def model_path(self) -> str:
        return os.path.join(self.models_dir, self.model_file)


class Config:
    """Main configuration container."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.auth = AuthConfig()
        self.server = ServerConfig()
        self.storage = StorageConfig()
    
    @classmethod
    def from_env(cls) -> 'Config':
        """
        Load configuration from environment variables.
        Useful for production deployment.
        """
        config = cls()
        
        # Server config from env
        config.server.host = os.getenv('SERVER_HOST', config.server.host)
        config.server.port = int(os.getenv('SERVER_PORT', config.server.port))
        config.server.debug = os.getenv('DEBUG', 'True').lower() == 'true'
        
        # Auth config from env
        config.auth.global_threshold = float(
            os.getenv('GLOBAL_THRESHOLD', config.auth.global_threshold)
        )
        
        return config
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'model': {
                'input_dim': self.model.input_dim,
                'hidden_dim1': self.model.hidden_dim1,
                'hidden_dim2': self.model.hidden_dim2,
                'embedding_dim': self.model.embedding_dim
            },
            'auth': {
                'enrollment_samples': self.auth.enrollment_samples,
                'global_threshold': self.auth.global_threshold,
                'user_threshold_samples': self.auth.user_threshold_samples
            },
            'storage': {
                'data_dir': self.storage.data_dir,
                'models_dir': self.storage.models_dir
            }
        }


# Global config instance
config = Config()