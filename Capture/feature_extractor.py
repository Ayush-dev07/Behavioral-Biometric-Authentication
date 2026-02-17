"""
Keystroke Feature Extraction Module
Processes raw keystroke timing data into feature vectors for neural network
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json


class KeystrokeFeatureExtractor:
    """
    Extracts timing features from keystroke sequences.
    
    Features extracted per keystroke pair:
    1. Dwell Time (DT): How long a key is held down
    2. Flight Time (FT): Time between releasing one key and pressing next
    3. Press-Press Time (PP): Time between consecutive key presses
    4. Release-Release Time (RR): Time between consecutive key releases
    """
    
    FEATURE_NAMES = ['dwell_time', 'flight_time', 'press_press', 'release_release']
    
    @staticmethod
    def extract_features(keystroke_data: List[Dict]) -> np.ndarray:
        """
        Extract timing features from raw keystroke data.
        
        Args:
            keystroke_data: List of keystroke events, each containing:
                {
                    'key': str,           # Character typed
                    'press_time': float,  # Timestamp when key pressed (ms)
                    'release_time': float # Timestamp when key released (ms)
                }
        
        Returns:
            features: numpy array of shape (n_pairs, 4)
                     where n_pairs = len(keystroke_data) - 1
                     Each row: [dwell_time, flight_time, pp_time, rr_time]
        
        Example:
            Input: [
                {'key': 'H', 'press_time': 1000, 'release_time': 1050},
                {'key': 'e', 'press_time': 1120, 'release_time': 1170}
            ]
            Output: [[50, 70, 120, 120]]
        """
        if len(keystroke_data) < 2:
            raise ValueError(f"Need at least 2 keystrokes, got {len(keystroke_data)}")
        
        features = []
        
        for i in range(len(keystroke_data) - 1):
            current = keystroke_data[i]
            next_key = keystroke_data[i + 1]
            
            # Feature 1: Dwell Time - how long current key is held
            dwell_time = current['release_time'] - current['press_time']
            
            # Feature 2: Flight Time - gap between releasing current and pressing next
            flight_time = next_key['press_time'] - current['release_time']
            
            # Feature 3: Press-Press Time - time between consecutive presses
            pp_time = next_key['press_time'] - current['press_time']
            
            # Feature 4: Release-Release Time - time between consecutive releases
            rr_time = next_key['release_time'] - current['release_time']
            
            features.append([dwell_time, flight_time, pp_time, rr_time])
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def extract_features_detailed(keystroke_data: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Extract features with detailed breakdown for analysis.
        
        Returns:
            Dict with separate arrays for each feature type
        """
        features = KeystrokeFeatureExtractor.extract_features(keystroke_data)
        
        return {
            'dwell_time': features[:, 0],
            'flight_time': features[:, 1],
            'press_press': features[:, 2],
            'release_release': features[:, 3],
            'all_features': features
        }
    
    @staticmethod
    def validate_keystroke_data(keystroke_data: List[Dict]) -> Tuple[bool, str]:
        """
        Validate keystroke data structure and timing.
        
        Args:
            keystroke_data: Keystroke events to validate
        
        Returns:
            valid: True if valid, False otherwise
            message: Error message if invalid, empty string if valid
        """
        if not keystroke_data:
            return False, "Empty keystroke data"
        
        if len(keystroke_data) < 2:
            return False, f"Need at least 2 keystrokes, got {len(keystroke_data)}"
        
        for i, ks in enumerate(keystroke_data):
            # Check required fields
            required_fields = ['key', 'press_time', 'release_time']
            missing_fields = [f for f in required_fields if f not in ks]
            if missing_fields:
                return False, f"Keystroke {i} missing fields: {missing_fields}"
            
            # Check that release comes after press
            if ks['release_time'] <= ks['press_time']:
                return False, f"Keystroke {i}: release_time must be > press_time"
            
            # Check chronological order
            if i > 0:
                prev = keystroke_data[i - 1]
                if ks['press_time'] < prev['press_time']:
                    return False, f"Keystroke {i}: not in chronological order"
        
        return True, ""
    
    @staticmethod
    def visualize_timing(keystroke_data: List[Dict]) -> str:
        """
        Create ASCII visualization of keystroke timing.
        
        Returns:
            visualization: String representation of timing
        """
        if not keystroke_data:
            return "No data"
        
        # Find time range
        start_time = keystroke_data[0]['press_time']
        end_time = max(ks['release_time'] for ks in keystroke_data)
        duration = end_time - start_time
        
        # Scale factor (characters per ms)
        scale = 100 / duration if duration > 0 else 1
        
        viz_lines = []
        viz_lines.append("Keystroke Timing Visualization:")
        viz_lines.append("=" * 100)
        
        for ks in keystroke_data:
            press_pos = int((ks['press_time'] - start_time) * scale)
            release_pos = int((ks['release_time'] - start_time) * scale)
            
            line = [' '] * 100
            line[press_pos] = '['
            for i in range(press_pos + 1, min(release_pos, 100)):
                line[i] = '='
            if release_pos < 100:
                line[release_pos] = ']'
            
            viz_lines.append(f"{ks['key']}: {''.join(line)}")
        
        viz_lines.append("=" * 100)
        viz_lines.append(f"Duration: {duration:.1f} ms")
        
        return '\n'.join(viz_lines)


class FeatureNormalizer:
    """
    Normalizes keystroke features using z-score normalization.
    Stores normalization parameters for consistent processing.
    """
    
    def __init__(self):
        """Initialize normalizer."""
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def fit(self, features: np.ndarray) -> 'FeatureNormalizer':
        """
        Fit normalizer to training data.
        
        Args:
            features: Feature array of shape (n_samples, n_features) or
                     (n_sequences, sequence_length, n_features)
        
        Returns:
            self: For method chaining
        """
        # Flatten to 2D if needed
        if features.ndim == 3:
            original_shape = features.shape
            features = features.reshape(-1, features.shape[-1])
        
        # Calculate statistics
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        
        # Avoid division by zero
        self.std = np.where(self.std == 0, 1.0, self.std)
        
        self.is_fitted = True
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using fitted parameters.
        
        Args:
            features: Features to normalize
        
        Returns:
            normalized: Normalized features
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Handle different input shapes
        original_shape = features.shape
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
        
        # Normalize
        normalized = (features - self.mean) / self.std
        
        # Restore original shape
        if len(original_shape) == 3:
            normalized = normalized.reshape(original_shape)
        
        return normalized.astype(np.float32)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            features: Features to fit and normalize
        
        Returns:
            normalized: Normalized features
        """
        return self.fit(features).transform(features)
    
    def inverse_transform(self, normalized_features: np.ndarray) -> np.ndarray:
        """
        Convert normalized features back to original scale.
        
        Args:
            normalized_features: Normalized features
        
        Returns:
            original: Features in original scale
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted.")
        
        return (normalized_features * self.std) + self.mean
    
    def save(self, filepath: str):
        """Save normalization parameters."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted normalizer")
        
        np.savez(
            filepath,
            mean=self.mean,
            std=self.std,
            is_fitted=self.is_fitted
        )
    
    def load(self, filepath: str):
        """Load normalization parameters."""
        data = np.load(filepath)
        self.mean = data['mean']
        self.std = data['std']
        self.is_fitted = bool(data['is_fitted'])


class DataAugmenter:
    """
    Augments keystroke data by adding realistic noise.
    Useful for expanding small enrollment datasets.
    """
    
    @staticmethod
    def augment_features(features: np.ndarray, 
                        noise_level: float = 0.05,
                        num_augmentations: int = 1) -> List[np.ndarray]:
        """
        Augment features by adding Gaussian noise.
        
        Args:
            features: Original features (seq_len, n_features)
            noise_level: Standard deviation of noise relative to feature values
            num_augmentations: Number of augmented samples to generate
        
        Returns:
            augmented: List of augmented feature arrays
        """
        augmented = []
        
        for _ in range(num_augmentations):
            # Add multiplicative noise
            noise = np.random.normal(1.0, noise_level, features.shape)
            augmented_sample = features * noise
            
            # Ensure no negative values (timing can't be negative)
            augmented_sample = np.maximum(augmented_sample, 0.0)
            
            augmented.append(augmented_sample.astype(np.float32))
        
        return augmented
    
    @staticmethod
    def augment_keystroke_data(keystroke_data: List[Dict],
                               timing_variance: float = 10.0,
                               num_augmentations: int = 1) -> List[List[Dict]]:
        """
        Augment raw keystroke data by adding timing variations.
        
        Args:
            keystroke_data: Original keystroke events
            timing_variance: Variance of timing noise (ms)
            num_augmentations: Number of augmented samples
        
        Returns:
            augmented: List of augmented keystroke sequences
        """
        augmented = []
        
        for _ in range(num_augmentations):
            new_sequence = []
            
            for ks in keystroke_data:
                # Add noise to press and release times
                press_noise = np.random.normal(0, timing_variance)
                release_noise = np.random.normal(0, timing_variance)
                
                new_ks = {
                    'key': ks['key'],
                    'press_time': ks['press_time'] + press_noise,
                    'release_time': ks['release_time'] + release_noise
                }
                
                # Ensure release is still after press
                if new_ks['release_time'] <= new_ks['press_time']:
                    new_ks['release_time'] = new_ks['press_time'] + 10
                
                new_sequence.append(new_ks)
            
            # Sort by press time to maintain order
            new_sequence.sort(key=lambda x: x['press_time'])
            
            augmented.append(new_sequence)
        
        return augmented


class EnrollmentProcessor:
    """
    Processes enrollment data (3 initial password samples).
    Handles feature extraction, normalization, and optional augmentation.
    """
    
    def __init__(self, augment: bool = False, augmentation_factor: int = 2):
        """
        Initialize enrollment processor.
        
        Args:
            augment: Whether to augment enrollment data
            augmentation_factor: Number of augmented samples per original
        """
        self.augment = augment
        self.augmentation_factor = augmentation_factor
        self.normalizer = FeatureNormalizer()
        self.feature_extractor = KeystrokeFeatureExtractor()
        self.augmenter = DataAugmenter()
    
    def process_enrollment(self, 
                          keystroke_samples: List[List[Dict]]) -> Tuple[List[np.ndarray], FeatureNormalizer]:
        """
        Process enrollment samples (typically 3 samples).
        
        Args:
            keystroke_samples: List of 3 keystroke sequences
        
        Returns:
            features_list: List of normalized feature arrays
            normalizer: Fitted normalizer for this user
        
        Raises:
            ValueError: If samples are invalid
        """
        if len(keystroke_samples) != 3:
            raise ValueError(f"Expected 3 enrollment samples, got {len(keystroke_samples)}")
        
        # Validate all samples
        for i, sample in enumerate(keystroke_samples):
            valid, msg = self.feature_extractor.validate_keystroke_data(sample)
            if not valid:
                raise ValueError(f"Sample {i+1} invalid: {msg}")
        
        # Extract features from all samples
        features_list = []
        for sample in keystroke_samples:
            features = self.feature_extractor.extract_features(sample)
            features_list.append(features)
        
        # Augment if requested
        if self.augment:
            augmented_features = []
            for features in features_list:
                augmented = self.augmenter.augment_features(
                    features,
                    noise_level=0.05,
                    num_augmentations=self.augmentation_factor
                )
                augmented_features.extend(augmented)
            features_list.extend(augmented_features)
        
        # Stack all features for normalization
        all_features = np.vstack(features_list)
        
        # Fit normalizer
        self.normalizer.fit(all_features)
        
        # Normalize each sample
        normalized_features = [
            self.normalizer.transform(features)
            for features in features_list
        ]
        
        return normalized_features, self.normalizer
    
    def process_authentication(self,
                              keystroke_data: List[Dict],
                              normalizer: FeatureNormalizer) -> np.ndarray:
        """
        Process single authentication attempt.
        
        Args:
            keystroke_data: Single keystroke sequence
            normalizer: Pre-fitted normalizer from enrollment
        
        Returns:
            features: Normalized feature array
        """
        # Validate
        valid, msg = self.feature_extractor.validate_keystroke_data(keystroke_data)
        if not valid:
            raise ValueError(f"Invalid keystroke data: {msg}")
        
        # Extract features
        features = self.feature_extractor.extract_features(keystroke_data)
        
        # Normalize using provided normalizer
        normalized = normalizer.transform(features)
        
        return normalized


class FeatureAnalyzer:
    """
    Analyzes and visualizes keystroke features.
    Useful for debugging and understanding typing patterns.
    """
    
    @staticmethod
    def analyze_features(features: np.ndarray) -> Dict:
        """
        Calculate statistics for features.
        
        Args:
            features: Feature array (seq_len, 4) or (n_samples, seq_len, 4)
        
        Returns:
            stats: Dictionary with feature statistics
        """
        # Flatten if needed
        if features.ndim == 3:
            features_2d = features.reshape(-1, features.shape[-1])
        else:
            features_2d = features
        
        feature_names = KeystrokeFeatureExtractor.FEATURE_NAMES
        
        stats = {}
        for i, name in enumerate(feature_names):
            feature_values = features_2d[:, i]
            stats[name] = {
                'mean': float(np.mean(feature_values)),
                'std': float(np.std(feature_values)),
                'min': float(np.min(feature_values)),
                'max': float(np.max(feature_values)),
                'median': float(np.median(feature_values))
            }
        
        return stats
    
    @staticmethod
    def compare_samples(sample1: np.ndarray, sample2: np.ndarray) -> Dict:
        """
        Compare two keystroke samples feature-wise.
        
        Args:
            sample1: First sample features
            sample2: Second sample features
        
        Returns:
            comparison: Dictionary with comparison metrics
        """
        if sample1.shape != sample2.shape:
            raise ValueError("Samples must have same shape")
        
        feature_names = KeystrokeFeatureExtractor.FEATURE_NAMES
        
        comparison = {}
        for i, name in enumerate(feature_names):
            feat1 = sample1[:, i]
            feat2 = sample2[:, i]
            
            # Calculate differences
            abs_diff = np.abs(feat1 - feat2)
            rel_diff = abs_diff / (np.abs(feat1) + 1e-6)  # Avoid division by zero
            
            comparison[name] = {
                'mean_abs_diff': float(np.mean(abs_diff)),
                'mean_rel_diff': float(np.mean(rel_diff)),
                'max_abs_diff': float(np.max(abs_diff)),
                'correlation': float(np.corrcoef(feat1, feat2)[0, 1])
            }
        
        return comparison
    
    @staticmethod
    def print_analysis(stats: Dict, title: str = "Feature Analysis"):
        """Print formatted feature analysis."""
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")
        
        for feature_name, feature_stats in stats.items():
            print(f"\n{feature_name.upper().replace('_', ' ')}:")
            print(f"  Mean:   {feature_stats['mean']:8.2f} ms")
            print(f"  Std:    {feature_stats['std']:8.2f} ms")
            print(f"  Min:    {feature_stats['min']:8.2f} ms")
            print(f"  Max:    {feature_stats['max']:8.2f} ms")
            print(f"  Median: {feature_stats['median']:8.2f} ms")


# Utility functions
def load_keystroke_json(filepath: str) -> List[Dict]:
    """Load keystroke data from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_keystroke_json(keystroke_data: List[Dict], filepath: str):
    """Save keystroke data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(keystroke_data, f, indent=2)


def create_sample_keystroke_data(password: str, 
                                 base_speed: float = 100.0,
                                 variance: float = 20.0) -> List[Dict]:
    """
    Create synthetic keystroke data for testing.
    
    Args:
        password: Password to simulate
        base_speed: Base typing speed (ms between keys)
        variance: Random variance in timing
    
    Returns:
        keystroke_data: Simulated keystroke sequence
    """
    keystroke_data = []
    current_time = 1000.0
    
    for char in password:
        # Randomize timing
        press_time = current_time + np.random.uniform(-variance, variance)
        dwell = base_speed * 0.5 + np.random.uniform(-variance/2, variance/2)
        release_time = press_time + dwell
        
        keystroke_data.append({
            'key': char,
            'press_time': press_time,
            'release_time': release_time
        })
        
        # Flight time to next key
        flight = base_speed + np.random.uniform(-variance, variance)
        current_time = release_time + flight
    
    return keystroke_data


if __name__ == "__main__":
    """
    Demo and testing of feature extraction.
    """
    print("="*70)
    print("Keystroke Feature Extraction - Demo")
    print("="*70)
    
    # Create sample keystroke data
    password = "SecurePass123"
    print(f"\nPassword: {password}")
    print(f"Length: {len(password)} characters")
    
    # Simulate typing the password
    print("\nSimulating keystroke data...")
    keystroke_data = create_sample_keystroke_data(
        password,
        base_speed=100,
        variance=20
    )
    
    print(f"Generated {len(keystroke_data)} keystroke events")
    
    # Visualize timing
    extractor = KeystrokeFeatureExtractor()
    print(extractor.visualize_timing(keystroke_data))
    
    # Validate data
    print("\nValidating keystroke data...")
    valid, msg = extractor.validate_keystroke_data(keystroke_data)
    if valid:
        print("✓ Data is valid")
    else:
        print(f"✗ Data is invalid: {msg}")
    
    # Extract features
    print("\nExtracting features...")
    features = extractor.extract_features(keystroke_data)
    print(f"Feature array shape: {features.shape}")
    print(f"  - {features.shape[0]} keystroke pairs")
    print(f"  - {features.shape[1]} features per pair")
    
    # Analyze features
    print("\nFeature Analysis:")
    analyzer = FeatureAnalyzer()
    stats = analyzer.analyze_features(features)
    analyzer.print_analysis(stats)
    
    # Test normalization
    print("\n" + "="*70)
    print("Testing Normalization")
    print("="*70)
    
    normalizer = FeatureNormalizer()
    normalized = normalizer.fit_transform(features)
    
    print(f"\nOriginal features:")
    print(f"  Mean: {features.mean(axis=0)}")
    print(f"  Std:  {features.std(axis=0)}")
    
    print(f"\nNormalized features:")
    print(f"  Mean: {normalized.mean(axis=0)}")
    print(f"  Std:  {normalized.std(axis=0)}")
    
    # Test augmentation
    print("\n" + "="*70)
    print("Testing Data Augmentation")
    print("="*70)
    
    augmenter = DataAugmenter()
    augmented = augmenter.augment_features(features, noise_level=0.05, num_augmentations=3)
    
    print(f"\nGenerated {len(augmented)} augmented samples")
    print("Comparing original vs augmented:")
    
    for i, aug_sample in enumerate(augmented[:2], 1):
        comparison = analyzer.compare_samples(features, aug_sample)
        print(f"\nAugmented Sample {i}:")
        for feat_name, comp_stats in comparison.items():
            print(f"  {feat_name}: "
                  f"mean_diff={comp_stats['mean_abs_diff']:.2f}ms, "
                  f"corr={comp_stats['correlation']:.3f}")
    
    # Test enrollment processing
    print("\n" + "="*70)
    print("Testing Enrollment Processing")
    print("="*70)
    
    # Create 3 enrollment samples
    enrollment_samples = [
        create_sample_keystroke_data(password, base_speed=100, variance=15)
        for _ in range(3)
    ]
    
    print(f"\nCreated {len(enrollment_samples)} enrollment samples")
    
    processor = EnrollmentProcessor(augment=True, augmentation_factor=2)
    processed_features, user_normalizer = processor.process_enrollment(enrollment_samples)
    
    print(f"Processed features: {len(processed_features)} samples")
    print(f"Each sample shape: {processed_features[0].shape}")
    
    # Test authentication processing
    print("\n" + "="*70)
    print("Testing Authentication Processing")
    print("="*70)
    
    # Simulate new authentication attempt
    auth_sample = create_sample_keystroke_data(password, base_speed=105, variance=18)
    auth_features = processor.process_authentication(auth_sample, user_normalizer)
    
    print(f"Authentication features shape: {auth_features.shape}")
    
    # Compare with enrollment
    comparison = analyzer.compare_samples(processed_features[0], auth_features)
    print("\nComparison with first enrollment sample:")
    for feat_name, comp_stats in comparison.items():
        print(f"  {feat_name}: correlation = {comp_stats['correlation']:.3f}")
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)