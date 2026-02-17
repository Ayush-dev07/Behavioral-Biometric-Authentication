"""
Authentication Manager
Handles user enrollment, authentication, and threshold management.
"""

import torch
import pickle
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

from siamese import SiameseRNNTriplet, Evaluator, Trainer
from Capture.feature_extractor import EnrollmentProcessor, FeatureNormalizer
from config import config
from utils import hash_password, timing_decorator

logger = logging.getLogger(__name__)


class ThresholdManager:
    """
    Manages authentication thresholds with adaptive refinement.
    Starts with global threshold and transitions to user-specific.
    """
    
    def __init__(self, 
                 global_threshold: float = 0.75,
                 user_threshold_samples: int = 50,
                 min_threshold: float = 0.5):
        """
        Initialize threshold manager.
        
        Args:
            global_threshold: Initial threshold for all users
            user_threshold_samples: Samples needed for user-specific threshold
            min_threshold: Minimum allowed threshold
        """
        self.global_threshold = global_threshold
        self.user_threshold_samples = user_threshold_samples
        self.min_threshold = min_threshold
        
        self.user_thresholds: Dict[str, float] = {}
        self.user_sample_counts: Dict[str, int] = {}
        self.user_genuine_scores: Dict[str, List[float]] = {}
    
    def get_threshold(self, user_id: str) -> float:
        """Get current threshold for a user."""
        sample_count = self.user_sample_counts.get(user_id, 0)
        
        if sample_count < self.user_threshold_samples:
            return self.global_threshold
        else:
            return self.user_thresholds.get(user_id, self.global_threshold)
    
    def update(self, user_id: str, similarity_score: float, authenticated: bool) -> None:
        """
        Update threshold data after authentication attempt.
        
        Args:
            user_id: User identifier
            similarity_score: Similarity from authentication
            authenticated: Whether authentication succeeded
        """
        if user_id not in self.user_genuine_scores:
            self.user_genuine_scores[user_id] = []
            self.user_sample_counts[user_id] = 0
        
        if authenticated:
            # Store genuine score
            self.user_genuine_scores[user_id].append(similarity_score)
            self.user_sample_counts[user_id] += 1
            
            # Recalculate threshold if enough samples
            if self.user_sample_counts[user_id] >= self.user_threshold_samples:
                self._calculate_user_threshold(user_id)
    
    def _calculate_user_threshold(self, user_id: str) -> None:
        """
        Calculate user-specific threshold.
        Uses mean - 2*std to cover ~95% of genuine attempts.
        """
        scores = np.array(self.user_genuine_scores[user_id])
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Threshold = mean - 2*std, but not lower than min_threshold
        threshold = max(self.min_threshold, mean_score - 2 * std_score)
        
        self.user_thresholds[user_id] = threshold
        logger.info(f"Updated threshold for {user_id}: {threshold:.3f} "
                   f"(mean={mean_score:.3f}, std={std_score:.3f})")
    
    def get_status(self, user_id: str) -> Dict:
        """Get threshold status for a user."""
        sample_count = self.user_sample_counts.get(user_id, 0)
        threshold = self.get_threshold(user_id)
        using_user_threshold = sample_count >= self.user_threshold_samples
        
        return {
            'threshold': threshold,
            'sample_count': sample_count,
            'using_user_threshold': using_user_threshold,
            'samples_needed': max(0, self.user_threshold_samples - sample_count),
            'threshold_type': 'user_specific' if using_user_threshold else 'global'
        }
    
    def save(self, filepath: str) -> None:
        """Save threshold data to disk."""
        data = {
            'global_threshold': self.global_threshold,
            'user_threshold_samples': self.user_threshold_samples,
            'min_threshold': self.min_threshold,
            'user_thresholds': self.user_thresholds,
            'user_sample_counts': self.user_sample_counts,
            'user_genuine_scores': self.user_genuine_scores
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Thresholds saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load threshold data from disk."""
        if not os.path.exists(filepath):
            logger.warning(f"Threshold file not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.global_threshold = data['global_threshold']
        self.user_threshold_samples = data['user_threshold_samples']
        self.min_threshold = data['min_threshold']
        self.user_thresholds = data['user_thresholds']
        self.user_sample_counts = data['user_sample_counts']
        self.user_genuine_scores = data['user_genuine_scores']
        
        logger.info(f"Thresholds loaded from {filepath}")


class AuthenticationManager:
    """
    Main authentication manager.
    Coordinates enrollment, authentication, and model management.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize authentication manager.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        
        # Initialize components
        self.model = None
        self.enrollment_processor = EnrollmentProcessor(
            augment=config.auth.augment_enrollment,
            augmentation_factor=config.auth.augmentation_factor
        )
        self.threshold_manager = ThresholdManager(
            global_threshold=config.auth.global_threshold,
            user_threshold_samples=config.auth.user_threshold_samples,
            min_threshold=config.auth.min_threshold
        )
        
        # User database
        self.users: Dict[str, Dict] = {}
        self.user_passwords: Dict[str, str] = {}
        
        # Load existing data
        self._load_state()
        
        # Initialize or load model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize or load the neural network model."""
        model_path = config.storage.model_path
        
        self.model = SiameseRNNTriplet(
            input_dim=config.model.input_dim,
            hidden_dim1=config.model.hidden_dim1,
            hidden_dim2=config.model.hidden_dim2,
            embedding_dim=config.model.embedding_dim
        )
        
        if os.path.exists(model_path):
            try:
                self.model = Trainer.load_model(model_path, self.model, self.device)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.info("Using untrained model")
        else:
            logger.info("No existing model found, using untrained model")
        
        self.model.to(self.device)
        self.model.eval()
    
    @timing_decorator
    def enroll_user(self, 
                   user_id: str,
                   password: str,
                   keystroke_samples: List[List[Dict]]) -> Dict:
        """
        Enroll a new user.
        
        Args:
            user_id: Unique user identifier
            password: Plain text password (will be hashed)
            keystroke_samples: List of 3 keystroke sequences
        
        Returns:
            result: Enrollment result with status
        """
        try:
            # Check if user exists
            if user_id in self.users:
                return {
                    'success': False,
                    'message': 'User already enrolled',
                    'user_id': user_id
                }
            
            # Validate enrollment samples
            if len(keystroke_samples) != config.auth.enrollment_samples:
                return {
                    'success': False,
                    'message': f'Expected {config.auth.enrollment_samples} samples, '
                              f'got {len(keystroke_samples)}',
                    'user_id': user_id
                }
            
            # Process enrollment
            features_list, normalizer = self.enrollment_processor.process_enrollment(
                keystroke_samples
            )
            
            # Hash password
            password_hash = hash_password(password)
            
            # Store user data
            self.users[user_id] = {
                'enrolled_at': datetime.now().isoformat(),
                'features': features_list,
                'normalizer': normalizer,
                'sample_count': len(features_list),
                'password_length': len(password)
            }
            
            self.user_passwords[user_id] = password_hash
            
            # Save state
            self._save_state()
            
            logger.info(f"User {user_id} enrolled successfully")
            
            return {
                'success': True,
                'message': 'User enrolled successfully',
                'user_id': user_id,
                'enrolled_at': self.users[user_id]['enrolled_at'],
                'num_samples': len(features_list)
            }
            
        except Exception as e:
            logger.error(f"Enrollment failed for {user_id}: {e}")
            return {
                'success': False,
                'message': f'Enrollment failed: {str(e)}',
                'user_id': user_id
            }
    
    @timing_decorator
    def authenticate(self,
                    user_id: str,
                    password: str,
                    keystroke_data: List[Dict]) -> Dict:
        """
        Authenticate a user.
        
        Args:
            user_id: User identifier
            password: Password attempt
            keystroke_data: Keystroke sequence for this attempt
        
        Returns:
            result: Authentication result with metrics
        """
        try:
            # Check if user exists
            if user_id not in self.users:
                return {
                    'authenticated': False,
                    'message': 'User not found',
                    'user_id': user_id,
                    'reason': 'user_not_found'
                }
            
            # Verify password
            password_hash = hash_password(password)
            if password_hash != self.user_passwords[user_id]:
                return {
                    'authenticated': False,
                    'message': 'Incorrect password',
                    'user_id': user_id,
                    'reason': 'password_mismatch'
                }
            
            # Process keystroke data
            user_data = self.users[user_id]
            auth_features = self.enrollment_processor.process_authentication(
                keystroke_data,
                user_data['normalizer']
            )
            
            # Calculate similarities
            similarities = []
            for enrolled_sample in user_data['features']:
                similarity = Evaluator.predict_similarity(
                    self.model,
                    enrolled_sample,
                    auth_features,
                    self.device
                )
                similarities.append(similarity)
            
            avg_similarity = float(np.mean(similarities))
            max_similarity = float(np.max(similarities))
            
            # Get threshold
            threshold = self.threshold_manager.get_threshold(user_id)
            threshold_status = self.threshold_manager.get_status(user_id)
            
            # Authentication decision
            authenticated = avg_similarity >= threshold
            
            # Update threshold manager and user data
            self.threshold_manager.update(user_id, avg_similarity, authenticated)
            
            if authenticated:
                # Add sample to user's collection
                user_data['features'].append(auth_features)
                user_data['sample_count'] += 1
                self._save_state()
                
                logger.info(f"User {user_id} authenticated (score: {avg_similarity:.3f})")
            else:
                logger.warning(f"User {user_id} rejected (score: {avg_similarity:.3f}, "
                             f"threshold: {threshold:.3f})")
            
            return {
                'authenticated': authenticated,
                'user_id': user_id,
                'similarity_score': avg_similarity,
                'max_similarity': max_similarity,
                'threshold': threshold,
                'threshold_type': threshold_status['threshold_type'],
                'sample_count': user_data['sample_count'],
                'samples_until_user_threshold': threshold_status['samples_needed'],
                'message': 'Authentication successful' if authenticated 
                          else 'Authentication failed: typing pattern mismatch',
                'reason': 'success' if authenticated else 'pattern_mismatch'
            }
            
        except Exception as e:
            logger.error(f"Authentication error for {user_id}: {e}")
            return {
                'authenticated': False,
                'message': f'Authentication error: {str(e)}',
                'user_id': user_id,
                'reason': 'error'
            }
    
    def get_user_stats(self, user_id: str) -> Optional[Dict]:
        """Get statistics for a user."""
        if user_id not in self.users:
            return None
        
        user_data = self.users[user_id]
        threshold_status = self.threshold_manager.get_status(user_id)
        
        return {
            'user_id': user_id,
            'enrolled_at': user_data['enrolled_at'],
            'total_samples': user_data['sample_count'],
            'current_threshold': threshold_status['threshold'],
            'using_user_threshold': threshold_status['using_user_threshold'],
            'samples_until_user_threshold': threshold_status['samples_needed'],
            'threshold_type': threshold_status['threshold_type']
        }
    
    def get_system_stats(self) -> Dict:
        """Get overall system statistics."""
        total_users = len(self.users)
        users_with_custom = sum(
            1 for user_id in self.users
            if self.threshold_manager.get_status(user_id)['using_user_threshold']
        )
        
        return {
            'total_users': total_users,
            'users_with_custom_threshold': users_with_custom,
            'global_threshold': self.threshold_manager.global_threshold,
            'model_loaded': self.model is not None
        }
    
    def delete_user(self, user_id: str) -> Dict:
        """Delete a user."""
        if user_id not in self.users:
            return {
                'success': False,
                'message': 'User not found',
                'user_id': user_id
            }
        
        del self.users[user_id]
        del self.user_passwords[user_id]
        
        # Also remove from threshold manager
        if user_id in self.threshold_manager.user_thresholds:
            del self.threshold_manager.user_thresholds[user_id]
        if user_id in self.threshold_manager.user_sample_counts:
            del self.threshold_manager.user_sample_counts[user_id]
        if user_id in self.threshold_manager.user_genuine_scores:
            del self.threshold_manager.user_genuine_scores[user_id]
        
        self._save_state()
        
        logger.info(f"User {user_id} deleted")
        
        return {
            'success': True,
            'message': 'User deleted successfully',
            'user_id': user_id
        }
    
    def update_global_threshold(self, new_threshold: float) -> Dict:
        """Update the global threshold."""
        if not 0.0 <= new_threshold <= 1.0:
            return {
                'success': False,
                'message': 'Threshold must be between 0.0 and 1.0'
            }
        
        old_threshold = self.threshold_manager.global_threshold
        self.threshold_manager.global_threshold = new_threshold
        self._save_state()
        
        logger.info(f"Global threshold updated: {old_threshold:.3f} -> {new_threshold:.3f}")
        
        return {
            'success': True,
            'message': 'Global threshold updated',
            'old_threshold': old_threshold,
            'new_threshold': new_threshold
        }
    
    def _save_state(self) -> None:
        """Save all state to disk."""
        # Save users
        users_data = {
            'users': self.users,
            'user_passwords': self.user_passwords
        }
        
        with open(config.storage.users_path, 'wb') as f:
            pickle.dump(users_data, f)
        
        # Save thresholds
        self.threshold_manager.save(config.storage.thresholds_path)
        
        logger.debug("State saved to disk")
    
    def _load_state(self) -> None:
        """Load state from disk."""
        # Load users
        if os.path.exists(config.storage.users_path):
            try:
                with open(config.storage.users_path, 'rb') as f:
                    users_data = pickle.load(f)
                
                self.users = users_data['users']
                self.user_passwords = users_data['user_passwords']
                
                logger.info(f"Loaded {len(self.users)} users from disk")
            except Exception as e:
                logger.error(f"Failed to load users: {e}")
        
        # Load thresholds
        self.threshold_manager.load(config.storage.thresholds_path)