import os
import sys
import json
import time
import hashlib
import uuid
import platform
import getpass
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    sys.exit(
        "\n[ERROR] pynput not found.\n"
        "Install it with:  pip install pynput\n"
    )

TARGET_PASSWORD: str = ""         
OUTPUT_FILE:     str = "keystroke_dataset.jsonl"
SAMPLES_PER_RUN: int = 10         
MAX_BACKSPACES:  int = 20          
SESSION_ID:      str = str(uuid.uuid4())

MODIFIER_KEYS = {
    pynput_keyboard.Key.shift,
    pynput_keyboard.Key.shift_r,
    pynput_keyboard.Key.shift_l,
    pynput_keyboard.Key.ctrl,
    pynput_keyboard.Key.ctrl_r,
    pynput_keyboard.Key.ctrl_l,
    pynput_keyboard.Key.alt,
    pynput_keyboard.Key.alt_r,
    pynput_keyboard.Key.alt_l,
    pynput_keyboard.Key.cmd,
    pynput_keyboard.Key.cmd_r,
    pynput_keyboard.Key.cmd_l,
    pynput_keyboard.Key.caps_lock,
}

def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def key_identifier(key) -> str:
    try:
        return f"char:{key.char}"
    except AttributeError:
        return f"special:{key.name}"


def is_special_key(key) -> bool:
    try:
        _ = key.char
        return False
    except AttributeError:
        return True


def is_modifier(key) -> bool:
    return key in MODIFIER_KEYS

class KeyEvent:
    __slots__ = (
        "key_id", "is_special", "is_modifier",
        "press_ts", "release_ts",
        "dwell_time", "key_position",
    )

    def __init__(self, key_id: str, is_special: bool, is_mod: bool,
                 press_ts: float, key_position: int):
        self.key_id       = key_id
        self.is_special   = is_special
        self.is_modifier  = is_mod
        self.press_ts     = press_ts       
        self.release_ts   = None
        self.dwell_time   = None           
        self.key_position = key_position   

    def close(self, release_ts: float):
        self.release_ts = release_ts
        self.dwell_time = release_ts - self.press_ts

    def to_dict(self) -> dict:
        return {
            "key_id":       self.key_id,
            "is_special":   self.is_special,
            "is_modifier":  self.is_modifier,
            "key_position": self.key_position,
            "press_ts":     self.press_ts,
            "release_ts":   self.release_ts,
            "dwell_time":   self.dwell_time,
        }

def compute_bigrams(events: list[KeyEvent]) -> list[dict]:
    bigrams = []
    for i in range(len(events) - 1):
        a, b = events[i], events[i + 1]
        if a.release_ts is None or b.release_ts is None:
            continue
        bigrams.append({
            "from_key":    a.key_id,
            "to_key":      b.key_id,
            "from_pos":    a.key_position,
            "to_pos":      b.key_position,
            "DD":          b.press_ts   - a.press_ts,     
            "UU":          b.release_ts - a.release_ts,  
            "UD":          b.press_ts   - a.release_ts,   
            "DU":          b.release_ts - a.press_ts,    
        })
    return bigrams

def compute_sequence_features(events: list[KeyEvent],
                               bigrams: list[dict],
                               total_elapsed: float,
                               backspace_count: int,
                               password_length: int) -> dict:
    
    dwell_times  = [e.dwell_time for e in events if e.dwell_time is not None]
    flight_times = [b["UD"]      for b in bigrams]

    def safe_stats(vals):
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None, "median": None}
        return {
            "mean":   statistics.mean(vals),
            "std":    statistics.pstdev(vals),
            "min":    min(vals),
            "max":    max(vals),
            "median": statistics.median(vals),
        }

    chars_per_second = password_length / total_elapsed if total_elapsed > 0 else 0
    chars_per_minute = chars_per_second * 60

    rhythm_cv = (
        (statistics.pstdev(flight_times) / statistics.mean(flight_times))
        if len(flight_times) > 1 and statistics.mean(flight_times) != 0
        else None
    )

    overlaps = sum(1 for b in bigrams if b["UD"] < 0)

    return {
        "total_elapsed_s":    total_elapsed,
        "chars_per_second":   chars_per_second,
        "chars_per_minute":   chars_per_minute,
        "backspace_count":    backspace_count,
        "error_rate":         backspace_count / max(password_length, 1),
        "overlap_count":      overlaps,
        "overlap_ratio":      overlaps / max(len(bigrams), 1),
        "rhythm_cv":          rhythm_cv,
        "dwell_stats":        safe_stats(dwell_times),
        "flight_stats":       safe_stats(flight_times),
        "total_dwell_time":   sum(dwell_times) if dwell_times else None,
        "total_flight_time":  sum(flight_times) if flight_times else None,
        "hold_ratio":         (
            sum(dwell_times) / total_elapsed
            if dwell_times and total_elapsed > 0 else None
        ),
    }


def build_feature_vector(events: list[KeyEvent],
                          bigrams: list[dict]) -> list[Optional[float]]:
   
    dwell_vec  = [e.dwell_time  if e.dwell_time  is not None else -1.0 for e in events]
    flight_vec = [b["UD"]                                              for b in bigrams]
    dd_vec     = [b["DD"]                                              for b in bigrams]
    uu_vec     = [b["UU"]                                              for b in bigrams]
    du_vec     = [b["DU"]                                              for b in bigrams]
    return {
        "dwell_vector":  dwell_vec,
        "UD_vector":     flight_vec,
        "DD_vector":     dd_vec,
        "UU_vector":     uu_vec,
        "DU_vector":     du_vec,
        "vector_length": len(dwell_vec),
    }

def collect_metadata(user_label: str, sample_index: int,
                     password_hash: str) -> dict:
    return {
        "session_id":      SESSION_ID,
        "sample_index":    sample_index,
        "user_label":      user_label,
        "password_hash":   password_hash,
        "collection_ts":   datetime.now(timezone.utc).isoformat(),
        "platform":        platform.system(),
        "platform_ver":    platform.version(),
        "python_version":  platform.python_version(),
        "hostname_hash":   sha256_hash(platform.node()),  
    }

class KeystrokeCollector:

    def __init__(self, target_password: str):
        self.target     = target_password
        self.pw_hash    = sha256_hash(target_password)
        self.pw_len     = len(target_password)

        self._reset()

    def _reset(self):
        self.typed_chars:  list[str]      = []   
        self.events:       list[KeyEvent] = []   
        self._live:        dict           = {}   
        self.backspaces:   int            = 0
        self.start_ts:     Optional[float] = None
        self.end_ts:       Optional[float] = None
        self.done:         bool           = False
        self.aborted:      bool           = False
        self._modifier_held: set         = set()
        self._caps_lock_on: bool         = False  

    def _on_press(self, key):
        now = time.perf_counter()  

        if self.done or self.aborted:
            return False           

        if key == pynput_keyboard.Key.caps_lock:
            self._caps_lock_on = not self._caps_lock_on
            return

        if is_modifier(key):
            self._modifier_held.add(key)
            return

        kid = key_identifier(key)

        if key == pynput_keyboard.Key.enter:
            self.end_ts = now
            self.done = True
            return False

        if key == pynput_keyboard.Key.esc:
            self.aborted = True
            return False

        if key == pynput_keyboard.Key.backspace:
            self.backspaces += 1
            if self.typed_chars:
                self.typed_chars.pop()
                if self.events:
                    removed = self.events.pop()
                    self._live.pop(removed.key_id, None)
            return
        
        char = None
        try:
            char = key.char
        except AttributeError:
            return  

        if not char:
            return
        shift_held = any(k in self._modifier_held for k in [
            pynput_keyboard.Key.shift,
            pynput_keyboard.Key.shift_l,
            pynput_keyboard.Key.shift_r,
        ])

        should_be_upper = self._caps_lock_on ^ shift_held  

        if char.isalpha():
            if should_be_upper:
                char = char.upper()
            else:
                char = char.lower()

        if self.start_ts is None:
            self.start_ts = now

        position = len(self.typed_chars)
        self.typed_chars.append(char)

        ev = KeyEvent(
            key_id       = kid,
            is_special   = is_special_key(key),
            is_mod       = bool(self._modifier_held),
            press_ts     = now,
            key_position = position,
        )
        self.events.append(ev)
        self._live[kid] = ev

    def _on_release(self, key):
        now = time.perf_counter()
        kid = key_identifier(key)

        if key == pynput_keyboard.Key.caps_lock:
            return  

        if is_modifier(key):
            self._modifier_held.discard(key)
            return

        if kid in self._live:
            self._live[kid].close(now)
            del self._live[kid]

    def collect_one(self, sample_index: int,
                    user_label: str, verbose: bool = True) -> Optional[dict]:
        self._reset()

        if verbose:
            print(f"\n  [{sample_index+1}] Type the password and press Enter: ", end="", flush=True)

        try:
            with pynput_keyboard.Listener(
                on_press   = self._on_press,
                on_release = self._on_release,
            ) as listener:
                listener.join()
        except Exception:
            pass   
        finally:
            import select
            if select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.readline()

        if verbose:
            print() 

        if self.aborted:
            print("  ↩  Attempt aborted (ESC pressed). Skipping.")
            return None

        typed_str = "".join(self.typed_chars)

        if typed_str != self.target:
            print(f"  ✗  Wrong password entered. Not recording. (typed {len(typed_str)} chars)")
            return None

        if self.start_ts is None or self.end_ts is None:
            print("  ✗  Timing error. Skipping sample.")
            return None

        total_elapsed = self.end_ts - self.start_ts

        bigrams  = compute_bigrams(self.events)
        seq_feat = compute_sequence_features(
            self.events, bigrams, total_elapsed,
            self.backspaces, self.pw_len
        )
        feat_vec = build_feature_vector(self.events, bigrams)
        meta     = collect_metadata(user_label, sample_index, self.pw_hash)

        record = {
            "metadata":         meta,

            "keystroke_events": [e.to_dict() for e in self.events],

            "bigrams":          bigrams,

            "sequence_features": seq_feat,

            "feature_vectors":  feat_vec,

            "password_correct": True,
            "password_hash":    self.pw_hash,
            "password_length":  self.pw_len,

            "active_modifiers_on_entry": [],  
        }

        if verbose:
            print(f"  ✓  Recorded — {len(self.events)} keys, "
                  f"{total_elapsed*1000:.0f}ms total, "
                  f"{seq_feat['chars_per_minute']:.1f} CPM, "
                  f"{self.backspaces} backspace(s)")

        return record
    
def run_collection_session(
    target_password: str,
    user_label:      str,
    output_file:     str = OUTPUT_FILE,
    n_samples:       int = SAMPLES_PER_RUN,
    warmup:          int = 2,
):

    collector = KeystrokeCollector(target_password)
    out_path  = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  Keystroke Dynamics Collector")
    print("  Behavioral Authentication — Training Data")
    print("═" * 60)
    print(f"  User label    : {user_label}")
    print(f"  Session ID    : {SESSION_ID}")
    print(f"  Target samples: {n_samples}  (+ {warmup} warm-up)")
    print(f"  Output file   : {out_path.resolve()}")
    print(f"  Password hash : {sha256_hash(target_password)[:16]}…")
    print("─" * 60)
    print("  Controls: Enter = submit | ESC = skip attempt | Ctrl+C = quit")
    print("─" * 60)

    if warmup > 0:
        print(f"\n  ── Warm-up ({warmup} attempts, not saved) ──")
        wu_done = 0
        while wu_done < warmup:
            r = collector.collect_one(wu_done, user_label=user_label)
            if r is not None:
                wu_done += 1
        print("  Warm-up complete. Starting real collection.\n")

    saved = 0
    attempt = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        while saved < n_samples:
            record = collector.collect_one(attempt, user_label=user_label)
            attempt += 1
            if record is not None:
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                saved += 1
                print(f"  Progress: {saved}/{n_samples} saved")

    print("\n" + "═" * 60)
    print(f"  ✅  Session complete — {saved} samples saved to:")
    print(f"      {out_path.resolve()}")
    print("═" * 60 + "\n")

def inspect_dataset(jsonl_path: str):
    path = Path(jsonl_path)
    if not path.exists():
        print(f"File not found: {jsonl_path}")
        return

    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("Empty dataset.")
        return

    users    = {}
    pw_hashes = set()
    for r in records:
        uid = r["metadata"]["user_label"]
        users.setdefault(uid, []).append(r)
        pw_hashes.add(r["password_hash"])

    print("\n" + "═" * 60)
    print(f"  Dataset: {path.name}")
    print(f"  Total records  : {len(records)}")
    print(f"  Unique users   : {len(users)}")
    print(f"  Unique pw hashes: {len(pw_hashes)}")
    print("─" * 60)
    for uid, recs in sorted(users.items()):
        cpms = [r["sequence_features"]["chars_per_minute"] for r in recs]
        print(f"  {uid:20s}  samples={len(recs):3d}  "
              f"avg_CPM={statistics.mean(cpms):5.1f}  "
              f"std_CPM={statistics.pstdev(cpms):5.2f}")
    print("═" * 60 + "\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Keystroke Dynamics Collector for Behavioral Authentication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python keystroke_collector.py --user alice --samples 20 --warmup 3
        """,
    )
    parser.add_argument("--user",     type=str, default=None,
                        help="User identifier label (e.g. user_001)")
    parser.add_argument("--samples",  type=int, default=SAMPLES_PER_RUN,
                        help=f"Number of valid samples to collect (default {SAMPLES_PER_RUN})")
    parser.add_argument("--warmup",   type=int, default=2,
                        help="Warm-up attempts to discard (default 2)")
    parser.add_argument("--output",   type=str, default=OUTPUT_FILE,
                        help=f"Output .jsonl path (default {OUTPUT_FILE})")
    parser.add_argument("--password", type=str, default=None,
                        help="Target password (prompted securely if omitted)")
    parser.add_argument("--inspect",  type=str, default=None, metavar="FILE",
                        help="Print dataset summary and exit")

    args = parser.parse_args()

    if args.inspect:
        inspect_dataset(args.inspect)
        return

    user_label = args.user or input("Enter user label (e.g. user_001): ").strip()
    if not user_label:
        sys.exit("[ERROR] User label cannot be empty.")

    password = args.password
    if not password:
        password = getpass.getpass("Set target password (hidden): ")
    if not password:
        sys.exit("[ERROR] Password cannot be empty.")

    try:
        run_collection_session(
            target_password = password,
            user_label      = user_label,
            output_file     = args.output,
            n_samples       = args.samples,
            warmup          = args.warmup,
        )
    finally:
        password = None
        import select
        time.sleep(0.1)
        if select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.readline()

if __name__ == "__main__":
    main()