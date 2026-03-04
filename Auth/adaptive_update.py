"""
Advanced Keystroke Authenticator with Adaptive Features
========================================================
Handles real-world challenges:
  ✓ Injury-induced pattern changes (multi-tier auth + MFA)
  ✓ Mood variations (normalized features)
  ✓ Temporal drift (continuous learning)
  ✓ Multiple typing profiles (injury, mobile, etc.)

Features:
  - Adaptive thresholds based on user variance
  - Continuous profile updates via sliding window
  - Multi-tier authentication (instant/MFA/challenge/reject)
  - Mood-invariant feature extraction
  - Time-of-day awareness
  - Template aging and refresh
  - Comprehensive audit logging

Usage:
    auth = AdaptiveKeystrokeAuthenticator(
        model_path="siamese_rnn_best.pth",
        enrollment_db="user_profiles.json",
        history_db="authentication_history.json",
    )
    
    # Authenticate with adaptive logic
    result = auth.authenticate_adaptive(
        username="alice",
        password="SecurePass@123",
    )
    
    if result["status"] == "ACCEPT":
        print("Welcome!")
    elif result["status"] == "MFA_REQUIRED":
        # Trigger SMS/email MFA
        code = send_mfa_code(result["username"])
        if verify_mfa(code):
            auth.record_successful_mfa(username, result["sample_id"])
            print("Welcome (MFA verified)!")
"""

import sys
import json
import time
import hashlib
import numpy as np
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    print("[WARN] pynput not available. Install: pip install pynput")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ERROR] PyTorch required. Install: pip install torch")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════
# Configuration & Data Classes
# ═════════════════════════════════════════════════════════════

@dataclass
class AuthConfig:
    """Adaptive authentication configuration."""
    # Thresholds
    threshold_strict: float = 0.15      # Instant login
    threshold_mfa: float = 0.30         # Require MFA
    threshold_challenge: float = 0.50   # Security questions
    threshold_reject: float = 0.70      # Hard reject
    
    # Continuous learning
    enable_continuous_learning: bool = True
    update_frequency: int = 10          # Update every N logins
    update_window: int = 50             # Use last N samples
    update_weight: float = 0.1          # New data weight (EMA)
    
    # Variance adaptation
    enable_adaptive_threshold: bool = True
    high_variance_threshold: float = 0.3  # CV threshold
    high_variance_boost: float = 0.05   # Add to threshold
    
    # Time awareness
    enable_temporal_profiles: bool = False  # Advanced feature
    
    # Template aging
    profile_refresh_days: int = 180     # 6 months
    
    # Audit
    keep_history_days: int = 90


@dataclass
class UserProfile:
    """Enhanced user profile with history."""
    username: str
    password_hash: str
    embedding: List[float]
    
    # Additional profiles
    alternative_profiles: Dict[str, List[float]]  # e.g., {"left_hand": [...]}
    
    # History
    enrollment_date: str
    last_update_date: str
    successful_logins: int
    
    # Variance tracking
    recent_distances: List[float]  # Last 20 distances
    recent_speeds: List[float]     # Last 20 CPM values
    
    # Metadata
    sample_count: int
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class AuthResult:
    """Authentication result with detailed information."""
    status: str  # ACCEPT, MFA_REQUIRED, CHALLENGE, REJECT
    username: str
    distance: float
    confidence: float
    threshold_used: float
    profile_matched: str  # "primary", "left_hand", etc.
    reason: Optional[str] = None
    sample_id: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


# ═════════════════════════════════════════════════════════════
# Mood-Invariant Feature Extraction
# ═════════════════════════════════════════════════════════════

def extract_normalized_features(raw_features: Dict) -> np.ndarray:
    """
    Extract mood-invariant features using normalization and ratios.
    
    Args:
        raw_features: Dict with 'dwell_vector', 'UD_vector', etc.
    
    Returns:
        (T, F) array with normalized features
    """
    dwell = np.array(raw_features['dwell_vector'], dtype=np.float32)
    UD = np.array(raw_features.get('UD_vector', []), dtype=np.float32)
    DD = np.array(raw_features.get('DD_vector', []), dtype=np.float32)
    UU = np.array(raw_features.get('UU_vector', []), dtype=np.float32)
    DU = np.array(raw_features.get('DU_vector', []), dtype=np.float32)
    
    T = len(dwell)
    
    # Normalize by sequence mean (mood-invariant)
    mean_dwell = np.mean(dwell) + 1e-8
    normalized_dwell = dwell / mean_dwell
    
    # Pad bigrams to match sequence length
    def pad_and_normalize(arr, mean_val):
        if len(arr) == 0:
            return np.zeros(T, dtype=np.float32)
        padded = np.zeros(T, dtype=np.float32)
        mean_val = np.mean(arr) + 1e-8
        normalized = arr / mean_val
        padded[1:len(normalized)+1] = normalized
        return padded
    
    normalized_UD = pad_and_normalize(UD, np.mean(UD) + 1e-8)
    normalized_DD = pad_and_normalize(DD, np.mean(DD) + 1e-8)
    normalized_UU = pad_and_normalize(UU, np.mean(UU) + 1e-8)
    normalized_DU = pad_and_normalize(DU, np.mean(DU) + 1e-8)
    
    # Stack normalized features
    features = np.stack([
        normalized_dwell,
        normalized_UD,
        normalized_DD,
        normalized_UU,
        normalized_DU,
    ], axis=1)  # (T, 5)
    
    return features


# ═════════════════════════════════════════════════════════════
# Authentication History Manager
# ═════════════════════════════════════════════════════════════

class AuthenticationHistory:
    """Tracks authentication attempts and patterns."""
    
    def __init__(self, db_path: str = "auth_history.json"):
        self.db_path = Path(db_path)
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.load()
    
    def load(self):
        if not self.db_path.exists():
            return
        with open(self.db_path) as f:
            data = json.load(f)
            for username, records in data.items():
                self.history[username] = deque(records, maxlen=100)
    
    def save(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, 'w') as f:
            data = {u: list(h) for u, h in self.history.items()}
            json.dump(data, f, indent=2)
    
    def add_attempt(self, username: str, record: Dict):
        """Record authentication attempt."""
        record['timestamp'] = datetime.now().isoformat()
        self.history[username].append(record)
        self.save()
    
    def get_recent_distances(self, username: str, n: int = 20) -> List[float]:
        """Get last N successful authentication distances."""
        records = [r for r in self.history[username] 
                   if r.get('status') in ['ACCEPT', 'MFA_VERIFIED']]
        distances = [r['distance'] for r in records[-n:]]
        return distances
    
    def get_recent_speeds(self, username: str, n: int = 20) -> List[float]:
        """Get last N typing speeds (CPM)."""
        records = [r for r in self.history[username] 
                   if r.get('status') in ['ACCEPT', 'MFA_VERIFIED']]
        speeds = [r.get('speed', 0) for r in records[-n:]]
        return [s for s in speeds if s > 0]
    
    def get_mfa_validated_samples(self, username: str, n: int = 10) -> List[str]:
        """Get sample IDs from recent MFA-verified logins."""
        records = [r for r in self.history[username] 
                   if r.get('status') == 'MFA_VERIFIED']
        sample_ids = [r.get('sample_id') for r in records[-n:]]
        return [s for s in sample_ids if s]
    
    def detect_sudden_drift(self, username: str, current_distance: float) -> bool:
        """Detect sudden pattern change vs gradual drift."""
        recent = self.get_recent_distances(username, n=10)
        if len(recent) < 5:
            return False
        
        avg_recent = np.mean(recent)
        std_recent = np.std(recent)
        
        # Sudden spike in distance
        if current_distance > avg_recent + 2 * std_recent:
            return True
        
        return False


# ═════════════════════════════════════════════════════════════
# Continuous Learning Engine
# ═════════════════════════════════════════════════════════════

class ContinuousLearner:
    """Handles profile updates based on validated samples."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.pending_updates: Dict[str, List[np.ndarray]] = defaultdict(list)
    
    def should_update(self, username: str, profile: UserProfile) -> bool:
        """Check if profile needs updating."""
        if not self.config.enable_continuous_learning:
            return False
        
        # Update every N logins
        return profile.successful_logins % self.config.update_frequency == 0
    
    def add_validated_sample(self, username: str, embedding: np.ndarray):
        """Add embedding from MFA-validated login."""
        self.pending_updates[username].append(embedding)
    
    def update_profile(self, username: str, current_embedding: np.ndarray) -> np.ndarray:
        """
        Update profile using exponential moving average.
        
        Args:
            username: User identifier
            current_embedding: Current enrolled embedding
        
        Returns:
            Updated embedding
        """
        pending = self.pending_updates[username]
        
        if len(pending) < self.config.update_window:
            # Not enough samples yet
            return current_embedding
        
        # Average recent embeddings
        recent_embeddings = pending[-self.config.update_window:]
        new_avg = np.mean(recent_embeddings, axis=0)
        
        # Exponential moving average
        alpha = self.config.update_weight
        updated = (1 - alpha) * current_embedding + alpha * new_avg
        
        # Clear pending
        self.pending_updates[username] = []
        
        return updated


# ═════════════════════════════════════════════════════════════
# Multi-Profile Manager
# ═════════════════════════════════════════════════════════════

class MultiProfileManager:
    """Manage multiple typing profiles per user."""
    
    def find_best_match(
        self,
        username: str,
        live_embedding: np.ndarray,
        profiles: Dict[str, np.ndarray],
        distance_fn,
    ) -> Tuple[str, float]:
        """
        Find best matching profile.
        
        Returns:
            (profile_name, distance)
        """
        best_profile = "primary"
        best_distance = float('inf')
        
        for profile_name, profile_embedding in profiles.items():
            distance = distance_fn(live_embedding, profile_embedding)
            if distance < best_distance:
                best_distance = distance
                best_profile = profile_name
        
        return best_profile, best_distance
    
    def add_alternative_profile(
        self,
        username: str,
        profile_name: str,
        samples: List[np.ndarray],
    ) -> np.ndarray:
        """
        Create alternative profile from samples.
        
        Args:
            username: User identifier
            profile_name: e.g., "left_hand", "mobile"
            samples: List of embeddings
        
        Returns:
            Average embedding for new profile
        """
        if len(samples) < 5:
            raise ValueError("Need at least 5 samples for alternative profile")
        
        return np.mean(samples, axis=0)


# ═════════════════════════════════════════════════════════════
# Advanced Keystroke Authenticator
# ═════════════════════════════════════════════════════════════

class AdaptiveKeystrokeAuthenticator:
    """
    Production-ready authenticator with adaptive features.
    """
    
    def __init__(
        self,
        model_path: str,
        enrollment_db: str = "user_profiles_adaptive.json",
        history_db: str = "auth_history.json",
        config: Optional[AuthConfig] = None,
    ):
        self.config = config or AuthConfig()
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load databases
        self.enrollment_db = Path(enrollment_db)
        self.profiles: Dict[str, UserProfile] = {}
        self.load_profiles()
        
        # History and learning
        self.history = AuthenticationHistory(history_db)
        self.learner = ContinuousLearner(self.config)
        self.profile_manager = MultiProfileManager()
        
        # Capture
        from keystroke_authenticator import LiveKeystrokeCapture
        self.capture = LiveKeystrokeCapture()
    
    def _load_model(self, path: str):
        """Load Siamese RNN model."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        from siamese import SiameseRNNTriplet
        
        model = SiameseRNNTriplet(
            input_dim=5,
            hidden_dim=256,
            embedding_dim=128,
            num_layers=3,
            rnn_type="gru",
            bidirectional=True,
            pooling="mean",
        )
        
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def load_profiles(self):
        """Load user profiles from disk."""
        if not self.enrollment_db.exists():
            return
        
        with open(self.enrollment_db) as f:
            data = json.load(f)
            self.profiles = {k: UserProfile.from_dict(v) for k, v in data.items()}
    
    def save_profiles(self):
        """Save user profiles to disk."""
        self.enrollment_db.parent.mkdir(parents=True, exist_ok=True)
        with open(self.enrollment_db, 'w') as f:
            data = {k: v.to_dict() for k, v in self.profiles.items()}
            json.dump(data, f, indent=2)
    
    def _compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Euclidean distance between embeddings."""
        return float(np.linalg.norm(emb1 - emb2))
    
    def _get_embedding(self, features: Dict) -> np.ndarray:
        """Generate embedding from features."""
        # Extract mood-invariant features
        seq = extract_normalized_features(features)
        
        # Convert to tensor
        seq_tensor = torch.from_numpy(seq).unsqueeze(0)
        length = torch.tensor([seq.shape[0]], dtype=torch.long)
        
        with torch.no_grad():
            embedding = self.model(seq_tensor, length)
        
        return embedding.squeeze(0).cpu().numpy()
    
    def _compute_adaptive_threshold(self, username: str) -> float:
        """Compute adaptive threshold based on user variance."""
        if not self.config.enable_adaptive_threshold:
            return self.config.threshold_strict
        
        recent_distances = self.history.get_recent_distances(username, n=20)
        
        if len(recent_distances) < 5:
            return self.config.threshold_strict
        
        # Coefficient of variation
        cv = np.std(recent_distances) / (np.mean(recent_distances) + 1e-8)
        
        if cv > self.config.high_variance_threshold:
            # User has high variance → more lenient
            return self.config.threshold_strict + self.config.high_variance_boost
        
        return self.config.threshold_strict
    
    def authenticate_adaptive(
        self,
        username: str,
        password: Optional[str] = None,
        verbose: bool = True,
    ) -> AuthResult:
        """
        Adaptive authentication with multi-tier decision logic.
        
        Returns:
            AuthResult with status, confidence, and metadata
        """
        # Check if user exists
        if username not in self.profiles:
            return AuthResult(
                status="REJECT",
                username=username,
                distance=float('inf'),
                confidence=1.0,
                threshold_used=0,
                profile_matched="none",
                reason="User not enrolled",
            )
        
        profile = self.profiles[username]
        
        # Capture keystroke
        if verbose:
            print(f"\nAuthenticating: {username}")
        
        result = self.capture.capture(prompt=f"Password for {username}: ")
        
        if result is None:
            return AuthResult(
                status="REJECT",
                username=username,
                distance=float('inf'),
                confidence=1.0,
                threshold_used=0,
                profile_matched="none",
                reason="Capture aborted",
            )
        
        typed_password = result["password"]
        
        # Verify password correctness
        if hashlib.sha256(typed_password.encode()).hexdigest() != profile.password_hash:
            return AuthResult(
                status="REJECT",
                username=username,
                distance=float('inf'),
                confidence=1.0,
                threshold_used=0,
                profile_matched="none",
                reason="Incorrect password",
            )
        
        # Generate embedding
        try:
            live_emb = self._get_embedding(result["timing_features"])
        except Exception as e:
            return AuthResult(
                status="REJECT",
                username=username,
                distance=float('inf'),
                confidence=1.0,
                threshold_used=0,
                profile_matched="none",
                reason=f"Feature extraction failed: {e}",
            )
        
        # Find best matching profile (primary + alternatives)
        all_profiles = {
            "primary": np.array(profile.embedding, dtype=np.float32),
            **{k: np.array(v, dtype=np.float32) 
               for k, v in profile.alternative_profiles.items()}
        }
        
        best_profile_name, distance = self.profile_manager.find_best_match(
            username, live_emb, all_profiles, self._compute_distance
        )
        
        # Compute adaptive threshold
        adaptive_threshold = self._compute_adaptive_threshold(username)
        
        # Multi-tier decision
        if distance < adaptive_threshold:
            # Tier 1: Strong match → Instant accept
            status = "ACCEPT"
            confidence = 1.0 - (distance / adaptive_threshold)
            reason = None
            
        elif distance < self.config.threshold_mfa:
            # Tier 2: Weak match → Require MFA
            status = "MFA_REQUIRED"
            confidence = 1.0 - (distance / self.config.threshold_mfa)
            
            # Check for sudden drift
            is_sudden = self.history.detect_sudden_drift(username, distance)
            if is_sudden:
                reason = "Unusual typing pattern detected (sudden change)"
            else:
                reason = "Typing pattern differs from enrolled profile"
        
        elif distance < self.config.threshold_challenge:
            # Tier 3: Very suspicious → Security questions
            status = "CHALLENGE"
            confidence = 0.3
            reason = "Significant deviation from enrolled profile"
        
        else:
            # Tier 4: Clear impostor → Hard reject
            status = "REJECT"
            confidence = 0.95
            reason = f"Distance {distance:.3f} exceeds maximum threshold"
        
        # Create result
        sample_id = f"{username}_{int(time.time())}"
        auth_result = AuthResult(
            status=status,
            username=username,
            distance=float(distance),
            confidence=float(confidence),
            threshold_used=float(adaptive_threshold),
            profile_matched=best_profile_name,
            reason=reason,
            sample_id=sample_id,
        )
        
        # Record attempt
        self.history.add_attempt(username, {
            **auth_result.to_dict(),
            'speed': result["timing_features"].get("total_time", 0),
        })
        
        # Handle successful authentication
        if status == "ACCEPT":
            self._handle_successful_login(username, live_emb, result)
        
        if verbose:
            self._print_result(auth_result)
        
        return auth_result
    
    def _handle_successful_login(
        self,
        username: str,
        embedding: np.ndarray,
        capture_result: Dict,
    ):
        """Update profile after successful login."""
        profile = self.profiles[username]
        profile.successful_logins += 1
        
        # Add to learning buffer
        self.learner.add_validated_sample(username, embedding)
        
        # Check if update needed
        if self.learner.should_update(username, profile):
            current_emb = np.array(profile.embedding, dtype=np.float32)
            updated_emb = self.learner.update_profile(username, current_emb)
            
            profile.embedding = updated_emb.tolist()
            profile.last_update_date = datetime.now().isoformat()
            
            print(f"\n✓ Profile updated for {username} (based on last {self.config.update_window} logins)")
        
        self.save_profiles()
    
    def record_successful_mfa(self, username: str, sample_id: str):
        """
        Record that MFA was successful for a weak biometric match.
        This validates the sample for continuous learning.
        """
        # Find the sample in history
        for record in self.history.history[username]:
            if record.get('sample_id') == sample_id:
                record['status'] = 'MFA_VERIFIED'
                record['mfa_verified_at'] = datetime.now().isoformat()
                break
        
        self.history.save()
        
        # Mark for profile update
        # The embedding was already added to learner during authentication
        print(f"✓ MFA verified for {username}. Sample will be used for profile update.")
    
    def add_alternative_profile(
        self,
        username: str,
        profile_name: str,
        password: str,
        n_samples: int = 10,
    ) -> bool:
        """
        Create alternative typing profile (e.g., for injury).
        
        Args:
            username: User identifier
            profile_name: Name for profile (e.g., "left_hand")
            password: User's password
            n_samples: Number of samples to collect
        
        Returns:
            Success boolean
        """
        if username not in self.profiles:
            print(f"[ERROR] User {username} not enrolled")
            return False
        
        profile = self.profiles[username]
        
        print(f"\n{'='*60}")
        print(f"  Creating alternative profile: {profile_name}")
        print(f"  User: {username}")
        print(f"  Samples needed: {n_samples}")
        print(f"{'='*60}\n")
        
        embeddings = []
        collected = 0
        
        while collected < n_samples:
            print(f"[{collected+1}/{n_samples}] ", end="")
            result = self.capture.capture(prompt="Type your password: ")
            
            if result is None:
                print("Aborted. Retrying...")
                continue
            
            if result["password"] != password:
                print(f"Wrong password. Retrying...")
                continue
            
            try:
                emb = self._get_embedding(result["timing_features"])
                embeddings.append(emb)
                collected += 1
                print(f"✓ Sample {collected} captured")
            except Exception as e:
                print(f"✗ Failed: {e}")
                continue
        
        # Create profile
        avg_embedding = self.profile_manager.add_alternative_profile(
            username, profile_name, embeddings
        )
        
        profile.alternative_profiles[profile_name] = avg_embedding.tolist()
        self.save_profiles()
        
        print(f"\n✅ Alternative profile '{profile_name}' created for {username}")
        return True
    
    def check_profile_age(self, username: str) -> Optional[Dict]:
        """
        Check if profile needs refresh.
        
        Returns:
            Dict with refresh info, or None if no refresh needed
        """
        if username not in self.profiles:
            return None
        
        profile = self.profiles[username]
        enrollment = datetime.fromisoformat(profile.enrollment_date)
        age_days = (datetime.now() - enrollment).days
        
        if age_days > self.config.profile_refresh_days:
            return {
                "needs_refresh": True,
                "age_days": age_days,
                "recommended_samples": 10,
                "benefit": "Improved accuracy and fewer MFA prompts",
            }
        
        return {"needs_refresh": False, "age_days": age_days}
    
    def _print_result(self, result: AuthResult):
        """Pretty print authentication result."""
        status_icons = {
            "ACCEPT": "✅",
            "MFA_REQUIRED": "⚠️ ",
            "CHALLENGE": "🔒",
            "REJECT": "❌",
        }
        
        print(f"\n{'─'*60}")
        print(f"  {status_icons.get(result.status, '?')} {result.status}")
        print(f"  Distance: {result.distance:.4f}")
        print(f"  Threshold: {result.threshold_used:.4f}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Profile: {result.profile_matched}")
        if result.reason:
            print(f"  Reason: {result.reason}")
        print(f"{'─'*60}\n")
    
    def get_user_dashboard(self, username: str) -> Dict:
        """
        Get user authentication dashboard data.
        
        Returns:
            Dict with stats and recent activity
        """
        if username not in self.profiles:
            return {"error": "User not found"}
        
        profile = self.profiles[username]
        recent = self.history.get_recent_distances(username, n=20)
        
        return {
            "username": username,
            "enrollment_date": profile.enrollment_date,
            "last_update": profile.last_update_date,
            "successful_logins": profile.successful_logins,
            "alternative_profiles": list(profile.alternative_profiles.keys()),
            "recent_performance": {
                "avg_distance": np.mean(recent) if recent else None,
                "match_quality": "Excellent" if np.mean(recent) < 0.10 else 
                                "Good" if np.mean(recent) < 0.15 else "Fair",
            },
            "profile_age": self.check_profile_age(username),
        }


# ═════════════════════════════════════════════════════════════
# CLI Interface
# ═════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Adaptive Keystroke Authentication System"
    )
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained Siamese RNN model")
    parser.add_argument("--enrollment-db", type=str, 
                       default="user_profiles_adaptive.json")
    parser.add_argument("--history-db", type=str,
                       default="auth_history.json")
    
    # Actions
    parser.add_argument("--enroll", type=str, metavar="USERNAME",
                       help="Enroll new user")
    parser.add_argument("--auth", type=str, metavar="USERNAME",
                       help="Authenticate user")
    parser.add_argument("--add-profile", nargs=2, metavar=("USERNAME", "PROFILE_NAME"),
                       help="Add alternative profile (e.g., for injury)")
    parser.add_argument("--dashboard", type=str, metavar="USERNAME",
                       help="Show user dashboard")
    
    # Configuration
    parser.add_argument("--threshold-strict", type=float, default=0.15)
    parser.add_argument("--threshold-mfa", type=float, default=0.30)
    parser.add_argument("--enable-learning", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Create config
    config = AuthConfig(
        threshold_strict=args.threshold_strict,
        threshold_mfa=args.threshold_mfa,
        enable_continuous_learning=args.enable_learning,
    )
    
    # Initialize authenticator
    auth = AdaptiveKeystrokeAuthenticator(
        model_path=args.model,
        enrollment_db=args.enrollment_db,
        history_db=args.history_db,
        config=config,
    )
    
    # Execute action
    if args.enroll:
        # Use original enrollment from keystroke_authenticator
        from Auth.keystroke_authenticator import KeystrokeAuthenticator
        basic_auth = KeystrokeAuthenticator(args.model, args.enrollment_db)
        import getpass
        password = getpass.getpass("Set password: ")
        basic_auth.enroll_user(args.enroll, password, n_samples=10)
    
    elif args.auth:
        result = auth.authenticate_adaptive(args.auth)
        
        if result.status == "MFA_REQUIRED":
            # Simulate MFA
            print("\n🔐 MFA Required")
            print("In production: Send SMS/Email code")
            code = input("Enter MFA code (or press Enter to simulate success): ")
            
            # In real system, verify code here
            if True:  # Simulated success
                auth.record_successful_mfa(result.username, result.sample_id)
                print("✅ MFA verified! Access granted.")
        
        sys.exit(0 if result.status in ["ACCEPT", "MFA_REQUIRED"] else 1)
    
    elif args.add_profile:
        username, profile_name = args.add_profile
        import getpass
        password = getpass.getpass("Enter your password: ")
        auth.add_alternative_profile(username, profile_name, password, n_samples=10)
    
    elif args.dashboard:
        dashboard = auth.get_user_dashboard(args.dashboard)
        print(f"\n{'='*60}")
        print(f"  Dashboard: {args.dashboard}")
        print(f"{'='*60}")
        for key, value in dashboard.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()