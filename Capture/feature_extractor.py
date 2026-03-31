import numpy as np
from typing import List, Dict, Tuple, Optional
import json

class KeystrokeFeatureExtractor:
    
    FEATURE_NAMES = ['dwell_time', 'flight_time', 'press_press', 'release_release']
    
    @staticmethod
    def extract_features(keystroke_data: List[Dict]) -> np.ndarray:
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
        if not keystroke_data:
            return False, "Empty keystroke data"
        
        if len(keystroke_data) < 2:
            return False, f"Need at least 2 keystrokes, got {len(keystroke_data)}"
        
        for i, ks in enumerate(keystroke_data):
            required_fields = ['key', 'press_time', 'release_time']
            missing_fields = [f for f in required_fields if f not in ks]
            if missing_fields:
                return False, f"Keystroke {i} missing fields: {missing_fields}"
            
            if ks['release_time'] <= ks['press_time']:
                return False, f"Keystroke {i}: release_time must be > press_time"
            
            if i > 0:
                prev = keystroke_data[i - 1]
                if ks['press_time'] < prev['press_time']:
                    return False, f"Keystroke {i}: not in chronological order"
        
        return True, ""
    
    @staticmethod
    def visualize_timing(keystroke_data: List[Dict]) -> str:
        if not keystroke_data:
            return "No data"

        start_time = keystroke_data[0]['press_time']
        end_time = max(ks['release_time'] for ks in keystroke_data)
        duration = end_time - start_time

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
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def fit(self, features: np.ndarray) -> 'FeatureNormalizer':
        if features.ndim == 3:
            original_shape = features.shape
            features = features.reshape(-1, features.shape[-1])
        
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        
        self.std = np.where(self.std == 0, 1.0, self.std)
        
        self.is_fitted = True
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        original_shape = features.shape
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
        
        normalized = (features - self.mean) / self.std
        
        if len(original_shape) == 3:
            normalized = normalized.reshape(original_shape)
        
        return normalized.astype(np.float32)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)
    
    def inverse_transform(self, normalized_features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted.")
        
        return (normalized_features * self.std) + self.mean
    
    def save(self, filepath: str):
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted normalizer")
        
        np.savez(
            filepath,
            mean=self.mean,
            std=self.std,
            is_fitted=self.is_fitted
        )
    
    def load(self, filepath: str):
        data = np.load(filepath)
        self.mean = data['mean']
        self.std = data['std']
        self.is_fitted = bool(data['is_fitted'])


class DataAugmenter:
    @staticmethod
    def augment_features(features: np.ndarray, 
                        noise_level: float = 0.05,
                        num_augmentations: int = 1) -> List[np.ndarray]:
        augmented = []
        
        for _ in range(num_augmentations):
            noise = np.random.normal(1.0, noise_level, features.shape)
            augmented_sample = features * noise
            
            augmented_sample = np.maximum(augmented_sample, 0.0)
            
            augmented.append(augmented_sample.astype(np.float32))
        
        return augmented
    
    @staticmethod
    def augment_keystroke_data(keystroke_data: List[Dict],
                               timing_variance: float = 10.0,
                               num_augmentations: int = 1) -> List[List[Dict]]:
        augmented = []
        
        for _ in range(num_augmentations):
            new_sequence = []
            
            for ks in keystroke_data:
                press_noise = np.random.normal(0, timing_variance)
                release_noise = np.random.normal(0, timing_variance)
                
                new_ks = {
                    'key': ks['key'],
                    'press_time': ks['press_time'] + press_noise,
                    'release_time': ks['release_time'] + release_noise
                }
                
                if new_ks['release_time'] <= new_ks['press_time']:
                    new_ks['release_time'] = new_ks['press_time'] + 10
                
                new_sequence.append(new_ks)
            
            new_sequence.sort(key=lambda x: x['press_time'])
            
            augmented.append(new_sequence)
        
        return augmented

class EnrollmentProcessor:
    def __init__(self, augment: bool = False, augmentation_factor: int = 2):
        self.augment = augment
        self.augmentation_factor = augmentation_factor
        self.normalizer = FeatureNormalizer()
        self.feature_extractor = KeystrokeFeatureExtractor()
        self.augmenter = DataAugmenter()
    
    def process_enrollment(self, 
                          keystroke_samples: List[List[Dict]]) -> Tuple[List[np.ndarray], FeatureNormalizer]:

        if len(keystroke_samples) != 3:
            raise ValueError(f"Expected 3 enrollment samples, got {len(keystroke_samples)}")
        
        for i, sample in enumerate(keystroke_samples):
            valid, msg = self.feature_extractor.validate_keystroke_data(sample)
            if not valid:
                raise ValueError(f"Sample {i+1} invalid: {msg}")
        
        features_list = []
        for sample in keystroke_samples:
            features = self.feature_extractor.extract_features(sample)
            features_list.append(features)
        
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
        
        all_features = np.vstack(features_list)
        
        self.normalizer.fit(all_features)
        
        normalized_features = [
            self.normalizer.transform(features)
            for features in features_list
        ]
        
        return normalized_features, self.normalizer
    
    def process_authentication(self,
                              keystroke_data: List[Dict],
                              normalizer: FeatureNormalizer) -> np.ndarray:

        valid, msg = self.feature_extractor.validate_keystroke_data(keystroke_data)
        if not valid:
            raise ValueError(f"Invalid keystroke data: {msg}")
        
        features = self.feature_extractor.extract_features(keystroke_data)
        
        normalized = normalizer.transform(features)
        
        return normalized

class FeatureAnalyzer:
    @staticmethod
    def analyze_features(features: np.ndarray) -> Dict:

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
        if sample1.shape != sample2.shape:
            raise ValueError("Samples must have same shape")
        
        feature_names = KeystrokeFeatureExtractor.FEATURE_NAMES
        
        comparison = {}
        for i, name in enumerate(feature_names):
            feat1 = sample1[:, i]
            feat2 = sample2[:, i]
            
            abs_diff = np.abs(feat1 - feat2)
            rel_diff = abs_diff / (np.abs(feat1) + 1e-6) 
            
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

def load_keystroke_json(filepath: str) -> List[Dict]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_keystroke_json(keystroke_data: List[Dict], filepath: str):
    with open(filepath, 'w') as f:
        json.dump(keystroke_data, f, indent=2)


def create_sample_keystroke_data(password: str, 
                                 base_speed: float = 100.0,
                                 variance: float = 20.0) -> List[Dict]:
    keystroke_data = []
    current_time = 1000.0
    
    for char in password:
        press_time = current_time + np.random.uniform(-variance, variance)
        dwell = base_speed * 0.5 + np.random.uniform(-variance/2, variance/2)
        release_time = press_time + dwell
        
        keystroke_data.append({
            'key': char,
            'press_time': press_time,
            'release_time': release_time
        })
        
        flight = base_speed + np.random.uniform(-variance, variance)
        current_time = release_time + flight
    
    return keystroke_data
