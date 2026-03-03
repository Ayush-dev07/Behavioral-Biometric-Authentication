"""
Keystroke Biometric Authenticator
==================================
Real-time authentication using behavioral biometrics.

Authentication pipeline:
  1. User enters username + password
  2. Live keystroke capture (timing data)
  3. Feature extraction (dwell, flight time vectors)
  4. Siamese RNN embedding generation
  5. Distance comparison against enrolled user profile
  6. Accept/Reject decision based on threshold

Architecture:
  - Supports both "verification" (1:1 match) and "identification" (1:N search)
  - Uses triplet loss embedding distance (L2 or cosine)
  - Configurable security thresholds (FAR/FRR trade-off)
  - Optional multi-sample fusion for increased accuracy

Usage:
    # 1. Train your Siamese RNN first (separate script)
    # 2. Enroll users by collecting reference keystroke samples
    # 3. Run authentication:

    from keystroke_authenticator import KeystrokeAuthenticator

    auth = KeystrokeAuthenticator(
        model_path="siamese_rnn.pth",
        enrollment_db="user_profiles.json",
        distance_metric="euclidean",  # or "cosine"
        threshold=0.15,               # tune based on EER
    )

    result = auth.authenticate_user(
        username="alice",
        password="Pass@1234",  # will be hashed, not stored
    )

    if result["authenticated"]:
        print(f"Welcome {username}!")
    else:
        print(f"Authentication failed: {result['reason']}")
"""

import sys
import json
import time
import hashlib
import numpy as np
import unicodedata
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, asdict

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    sys.exit("Error: pip install pynput")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not found. Model loading disabled.")


# ─────────────────────────────────────────────────────────────
# Keystroke capture (simplified from collector)
# ─────────────────────────────────────────────────────────────

MODIFIER_KEYS = {
    pynput_keyboard.Key.shift, pynput_keyboard.Key.shift_r, pynput_keyboard.Key.shift_l,
    pynput_keyboard.Key.ctrl,  pynput_keyboard.Key.ctrl_r,  pynput_keyboard.Key.ctrl_l,
    pynput_keyboard.Key.alt,   pynput_keyboard.Key.alt_r,   pynput_keyboard.Key.alt_l,
    pynput_keyboard.Key.cmd,   pynput_keyboard.Key.cmd_r,   pynput_keyboard.Key.cmd_l,
}


def _extract_char(key) -> str | None:
    """Extract printable character from pynput key (handles all charsets)."""
    if isinstance(key, pynput_keyboard.Key):
        return None
    ch = getattr(key, "char", None)
    if ch is not None:
        if len(ch) == 1 and unicodedata.category(ch) not in ("Cc", "Cf", "Cs"):
            return ch
    vk = getattr(key, "vk", None)
    if vk is not None and 33 <= vk <= 126:
        return chr(vk)
    return None


@dataclass
class KeyEvent:
    """Single keystroke timing record."""
    position:    int
    char:        str
    press_ts:    float
    release_ts:  float
    dwell_time:  float

    @property
    def dwell_ms(self) -> float:
        return self.dwell_time * 1000


class LiveKeystrokeCapture:
    """
    Captures a single password entry with full timing data.
    Returns the typed string + timing features for authentication.
    """

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._reset()

    def _reset(self):
        self.typed_chars: list[str]       = []
        self.events:      list[KeyEvent]  = []
        self._live:       dict            = {}   # pos -> (press_ts, char)
        self.start_ts:    Optional[float] = None
        self.end_ts:      Optional[float] = None
        self.done:        bool            = False
        self.aborted:     bool            = False
        self._modifiers:  set             = set()

    def _on_press(self, key):
        now = time.perf_counter()

        if self.done or self.aborted:
            return False

        if key in MODIFIER_KEYS:
            self._modifiers.add(key)
            return

        if key == pynput_keyboard.Key.enter:
            self.end_ts = now
            self.done = True
            return False

        if key == pynput_keyboard.Key.esc:
            self.aborted = True
            return False

        if key == pynput_keyboard.Key.backspace:
            if self.typed_chars:
                self.typed_chars.pop()
                if self.events:
                    self.events.pop()
                pos = len(self.typed_chars)
                self._live.pop(pos, None)
            return

        char = _extract_char(key)
        if char is None:
            return

        if self.start_ts is None:
            self.start_ts = now

        pos = len(self.typed_chars)
        self.typed_chars.append(char)
        self._live[pos] = (now, char)

    def _on_release(self, key):
        now = time.perf_counter()

        if key in MODIFIER_KEYS:
            self._modifiers.discard(key)
            return

        char = _extract_char(key)
        if char is None:
            return

        # LIFO: find most recent open press of this char
        for pos in sorted(self._live.keys(), reverse=True):
            press_ts, c = self._live[pos]
            if c == char:
                ev = KeyEvent(
                    position   = pos,
                    char       = char,
                    press_ts   = press_ts,
                    release_ts = now,
                    dwell_time = now - press_ts,
                )
                self.events.append(ev)
                del self._live[pos]
                break

    def capture(self, prompt: str = "Enter password: ") -> Optional[dict]:
        """
        Block until user types password and presses Enter.
        Returns dict with 'password' (plaintext) and 'timing_features' or None if aborted.
        """
        self._reset()
        print(prompt, end="", flush=True)

        with pynput_keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release, suppress=True
        ) as listener:
            listener.join(timeout=self.timeout)

        print()  # newline after hidden input

        if self.aborted:
            return None

        if not self.done:
            print("[TIMEOUT] Authentication attempt timed out.")
            return None

        typed_str = "".join(self.typed_chars)

        # Sort events by position to ensure correct sequence
        self.events.sort(key=lambda e: e.position)

        return {
            "password": typed_str,
            "timing_features": self._extract_features(),
        }

    def _extract_features(self) -> dict:
        """
        Extract the same feature vectors used during training.
        Returns dict matching keystroke_dataset.py format.
        """
        if len(self.events) == 0:
            return {
                "dwell_vector": [],
                "UD_vector": [],
                "DD_vector": [],
                "UU_vector": [],
                "DU_vector": [],
                "sequence_length": 0,
            }

        # Per-key dwell times
        dwell_vec = [e.dwell_time for e in self.events]

        # Bigram flight times
        UD_vec, DD_vec, UU_vec, DU_vec = [], [], [], []
        for i in range(len(self.events) - 1):
            a, b = self.events[i], self.events[i + 1]
            DD_vec.append(b.press_ts   - a.press_ts)
            UU_vec.append(b.release_ts - a.release_ts)
            UD_vec.append(b.press_ts   - a.release_ts)  # flight time
            DU_vec.append(b.release_ts - a.press_ts)

        return {
            "dwell_vector":    dwell_vec,
            "UD_vector":       UD_vec,
            "DD_vector":       DD_vec,
            "UU_vector":       UU_vec,
            "DU_vector":       DU_vec,
            "sequence_length": len(dwell_vec),
            "total_time":      self.end_ts - self.start_ts if self.start_ts and self.end_ts else 0,
        }


# ─────────────────────────────────────────────────────────────
# Feature preprocessing for model input
# ─────────────────────────────────────────────────────────────

def normalize_features(features: dict, stats: dict) -> np.ndarray:
    """
    Convert raw timing dict to normalized (T, F) array for RNN input.
    Matches the training pipeline normalization.
    """
    dwell = np.array(features["dwell_vector"], dtype=np.float32).reshape(-1, 1)
    UD    = np.array(features["UD_vector"],    dtype=np.float32)

    # Pad UD to match dwell length (UD has N-1 elements for N keys)
    UD_padded = np.zeros(len(dwell), dtype=np.float32)
    UD_padded[1:len(UD)+1] = UD  # shift by 1: bigram i-1→i goes to position i

    seq = np.concatenate([dwell, UD_padded.reshape(-1, 1)], axis=1)  # (T, 2)

    # Normalize using dataset statistics
    if stats:
        mean = np.array(stats["mean"], dtype=np.float32)
        std  = np.array(stats["std"],  dtype=np.float32) + 1e-8
        seq = (seq - mean) / std

    return seq


# ─────────────────────────────────────────────────────────────
# Enrollment database (user profiles)
# ─────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    """Enrolled user's reference keystroke embedding."""
    username:       str
    password_hash:  str
    embedding:      list[float]  # averaged from enrollment samples
    sample_count:   int
    enrollment_date: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class EnrollmentDB:
    """Persistent storage of user keystroke profiles."""

    def __init__(self, db_path: str = "user_profiles.json"):
        self.db_path = Path(db_path)
        self.profiles: dict[str, UserProfile] = {}
        self.load()

    def load(self):
        if not self.db_path.exists():
            return
        with open(self.db_path, encoding="utf-8") as f:
            data = json.load(f)
            self.profiles = {
                k: UserProfile.from_dict(v) for k, v in data.items()
            }

    def save(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            data = {k: v.to_dict() for k, v in self.profiles.items()}
            json.dump(data, f, indent=2)

    def add_user(self, profile: UserProfile):
        self.profiles[profile.username] = profile
        self.save()

    def get_user(self, username: str) -> Optional[UserProfile]:
        return self.profiles.get(username)

    def user_exists(self, username: str) -> bool:
        return username in self.profiles


# ─────────────────────────────────────────────────────────────
# Distance metrics for embedding comparison
# ─────────────────────────────────────────────────────────────

def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """L2 distance between two embeddings."""
    return np.linalg.norm(emb1 - emb2)


def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine distance: 1 - cosine_similarity."""
    dot = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    return 1.0 - (dot / (norm + 1e-8))


DISTANCE_FUNCTIONS = {
    "euclidean": euclidean_distance,
    "cosine":    cosine_distance,
}


# ─────────────────────────────────────────────────────────────
# Main Authenticator
# ─────────────────────────────────────────────────────────────

class KeystrokeAuthenticator:
    """
    Complete keystroke biometric authentication system.

    Workflow:
      1. enroll(username, password) — collect samples, average embeddings
      2. authenticate(username, password) — verify live attempt against profile
    """

    def __init__(
        self,
        model_path:      str,
        enrollment_db:   str = "user_profiles.json",
        distance_metric: Literal["euclidean", "cosine"] = "euclidean",
        threshold:       float = 0.15,
        normalization_stats: Optional[dict] = None,
    ):
        """
        Args:
            model_path: Path to trained Siamese RNN .pth file
            enrollment_db: JSON file storing user embedding profiles
            distance_metric: "euclidean" or "cosine"
            threshold: Accept if distance < threshold (tune for FAR/FRR balance)
            normalization_stats: {"mean": [...], "std": [...]} from training data
        """
        self.model = self._load_model(model_path) if TORCH_AVAILABLE else None
        self.db = EnrollmentDB(enrollment_db)
        self.distance_fn = DISTANCE_FUNCTIONS[distance_metric]
        self.threshold = threshold
        self.norm_stats = normalization_stats or {"mean": [0, 0], "std": [1, 1]}
        self.capture = LiveKeystrokeCapture()

    def _load_model(self, path: str):
        """Load the trained Siamese RNN model."""
        if not Path(path).exists():
            print(f"[WARN] Model not found at {path}. Authenticator will fail.")
            return None

        try:
            # Import the model architecture
            from siamese import SiameseRNNTriplet

            # Initialize with default architecture (must match training config)
            model = SiameseRNNTriplet(
                input_dim=2,        # dwell + UD
                hidden_dim=128,
                embedding_dim=128,
                num_layers=2,
                rnn_type="gru",
                bidirectional=True,
                pooling="mean",
                dropout=0.3,
            )

            # Load trained weights
            state_dict = torch.load(path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()

            print(f"[INFO] Model loaded from {path}")
            return model

        except ImportError:
            print("[ERROR] siamese_rnn_model.py not found. Cannot load model.")
            print("        Make sure siamese_rnn_model.py is in the same directory.")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return None

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def _get_embedding(self, features: dict) -> np.ndarray:
        """Generate embedding from timing features using the trained model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot generate embedding.")

        # Normalize and convert to tensor
        seq = normalize_features(features, self.norm_stats)
        seq_tensor = torch.from_numpy(seq).unsqueeze(0)  # (1, T, F)
        length = torch.tensor([seq.shape[0]], dtype=torch.long)

        with torch.no_grad():
            embedding = self.model(seq_tensor, length)  # (1, emb_dim)

        return embedding.squeeze(0).cpu().numpy()

    # ── Enrollment ──────────────────────────────────────────────

    def enroll_user(
        self,
        username: str,
        password: str,
        n_samples: int = 5,
        verbose: bool = True,
    ) -> bool:
        """
        Enroll a new user by collecting multiple keystroke samples.

        Args:
            username: Unique identifier
            password: The password they will use (stored as hash only)
            n_samples: Number of typing samples to collect (≥3 recommended)
            verbose: Print progress

        Returns:
            True if enrollment succeeded, False otherwise
        """
        if self.db.user_exists(username):
            print(f"[ERROR] User '{username}' already enrolled.")
            return False

        pw_hash = self._hash_password(password)

        print(f"\n{'='*60}")
        print(f"  Enrolling user: {username}")
        print(f"  Collecting {n_samples} keystroke samples...")
        print(f"{'='*60}\n")

        embeddings = []
        collected = 0

        while collected < n_samples:
            attempt = collected + 1
            print(f"[{attempt}/{n_samples}] ", end="")

            result = self.capture.capture(prompt="Type your password: ")

            if result is None:
                print("Aborted. Retrying...")
                continue

            if result["password"] != password:
                print(f"Wrong password. Expected {len(password)} chars, got {len(result['password'])}. Retrying...")
                continue

            try:
                emb = self._get_embedding(result["timing_features"])
                embeddings.append(emb)
                collected += 1
                if verbose:
                    print(f"✓ Sample {collected} captured (emb dim: {len(emb)})")
            except Exception as e:
                print(f"✗ Feature extraction failed: {e}")
                continue

        # Average embeddings across samples
        avg_embedding = np.mean(embeddings, axis=0).tolist()

        profile = UserProfile(
            username       = username,
            password_hash  = pw_hash,
            embedding      = avg_embedding,
            sample_count   = n_samples,
            enrollment_date = time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        self.db.add_user(profile)

        print(f"\n{'='*60}")
        print(f"  ✅ User '{username}' enrolled successfully!")
        print(f"  Embedding dimension: {len(avg_embedding)}")
        print(f"  Samples collected: {n_samples}")
        print(f"{'='*60}\n")

        return True

    # ── Authentication ──────────────────────────────────────────

    def authenticate_user(
        self,
        username: str,
        password: Optional[str] = None,  # if None, will prompt
        verbose: bool = True,
    ) -> dict:
        """
        Authenticate a user via live keystroke capture.

        Returns:
            {
                "authenticated": bool,
                "username": str,
                "distance": float,
                "threshold": float,
                "reason": str,  # if failed
            }
        """
        # Check if user exists
        profile = self.db.get_user(username)
        if profile is None:
            return {
                "authenticated": False,
                "username": username,
                "distance": None,
                "threshold": self.threshold,
                "reason": f"User '{username}' not found in enrollment database.",
            }

        # Capture live keystroke
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Authenticating user: {username}")
            print(f"{'─'*60}\n")

        result = self.capture.capture(prompt=f"Password for {username}: ")

        if result is None:
            return {
                "authenticated": False,
                "username": username,
                "distance": None,
                "threshold": self.threshold,
                "reason": "Authentication aborted or timed out.",
            }

        typed_password = result["password"]

        # Verify password correctness first
        if self._hash_password(typed_password) != profile.password_hash:
            return {
                "authenticated": False,
                "username": username,
                "distance": None,
                "threshold": self.threshold,
                "reason": "Incorrect password.",
            }

        # Generate embedding from live keystroke
        try:
            live_emb = self._get_embedding(result["timing_features"])
        except Exception as e:
            return {
                "authenticated": False,
                "username": username,
                "distance": None,
                "threshold": self.threshold,
                "reason": f"Feature extraction failed: {e}",
            }

        # Compute distance to enrolled profile
        ref_emb = np.array(profile.embedding, dtype=np.float32)
        distance = self.distance_fn(live_emb, ref_emb)

        authenticated = distance < self.threshold

        if verbose:
            print(f"  Distance: {distance:.4f}")
            print(f"  Threshold: {self.threshold:.4f}")
            print(f"  Result: {'✅ ACCEPTED' if authenticated else '❌ REJECTED'}")
            print(f"{'─'*60}\n")

        return {
            "authenticated": authenticated,
            "username": username,
            "distance": float(distance),
            "threshold": self.threshold,
            "reason": None if authenticated else f"Distance {distance:.4f} exceeds threshold {self.threshold:.4f}",
        }

    # ── Identification (1:N search) ─────────────────────────────

    def identify_user(self, password: str, top_k: int = 3) -> list[dict]:
        """
        Identification mode: find the most likely user(s) from all enrolled profiles.

        Returns list of matches sorted by distance (closest first), up to top_k.
        """
        result = self.capture.capture(prompt="Enter password: ")
        if result is None or result["password"] != password:
            return []

        live_emb = self._get_embedding(result["timing_features"])

        candidates = []
        for username, profile in self.db.profiles.items():
            if self._hash_password(password) != profile.password_hash:
                continue  # password doesn't match this user

            ref_emb = np.array(profile.embedding, dtype=np.float32)
            distance = self.distance_fn(live_emb, ref_emb)

            candidates.append({
                "username": username,
                "distance": float(distance),
                "accepted": distance < self.threshold,
            })

        candidates.sort(key=lambda x: x["distance"])
        return candidates[:top_k]


# ─────────────────────────────────────────────────────────────
# CLI interface
# ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Keystroke Biometric Authenticator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll a new user
  python keystroke_authenticator.py --enroll alice --model siamese_rnn.pth --samples 5

  # Authenticate an existing user
  python keystroke_authenticator.py --auth alice --model siamese_rnn.pth

  # Identification mode (1:N search)
  python keystroke_authenticator.py --identify --model siamese_rnn.pth
        """,
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained Siamese RNN model (.pth)")
    parser.add_argument("--db", type=str, default="user_profiles.json",
                        help="User enrollment database")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Distance threshold for acceptance")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")

    # Modes
    parser.add_argument("--enroll", type=str, metavar="USERNAME",
                        help="Enroll a new user")
    parser.add_argument("--auth", type=str, metavar="USERNAME",
                        help="Authenticate an existing user")
    parser.add_argument("--identify", action="store_true",
                        help="Identification mode (1:N)")

    parser.add_argument("--samples", type=int, default=5,
                        help="Number of enrollment samples (default 5)")
    parser.add_argument("--password", type=str, default=None,
                        help="Password (prompted if omitted)")

    args = parser.parse_args()

    # Initialize authenticator
    auth = KeystrokeAuthenticator(
        model_path      = args.model,
        enrollment_db   = args.db,
        distance_metric = args.metric,
        threshold       = args.threshold,
    )

    # ── Enrollment mode ──
    if args.enroll:
        import getpass
        password = args.password or getpass.getpass("Set password: ")
        success = auth.enroll_user(args.enroll, password, n_samples=args.samples)
        sys.exit(0 if success else 1)

    # ── Authentication mode ──
    elif args.auth:
        result = auth.authenticate_user(args.auth)
        if result["authenticated"]:
            print(f"✅ Welcome, {args.auth}!")
            sys.exit(0)
        else:
            print(f"❌ Authentication failed: {result['reason']}")
            sys.exit(1)

    # ── Identification mode ──
    elif args.identify:
        import getpass
        password = args.password or getpass.getpass("Enter password: ")
        matches = auth.identify_user(password, top_k=3)

        if not matches:
            print("No matches found.")
            sys.exit(1)

        print("\nTop matches:")
        for i, m in enumerate(matches, 1):
            status = "✅" if m["accepted"] else "❌"
            print(f"  {i}. {m['username']:15s}  distance={m['distance']:.4f}  {status}")
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()