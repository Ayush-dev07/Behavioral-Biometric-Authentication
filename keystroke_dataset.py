import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not found. KeystrokeDataset will return numpy arrays only.")

FEATURE_SCHEMA = {
    "key_id":         "str  — e.g. 'char:a', 'special:shift'",
    "is_special":     "bool — True if function/control key",
    "is_modifier":    "bool — True if shift/ctrl/alt/cmd",
    "key_position":   "int  — 0-indexed position in the typed sequence",
    "press_ts":       "float— perf_counter epoch at key-down (seconds)",
    "release_ts":     "float— perf_counter epoch at key-up  (seconds)",
    "dwell_time":     "float— release_ts - press_ts (seconds); -1 if unreleased",

    "from_key":       "str  — key_id of the preceding key",
    "to_key":         "str  — key_id of the following key",
    "from_pos":       "int  — position of the preceding key",
    "to_pos":         "int  — position of the following key",
    "DD":             "float— down→down latency (inter-keydown)",
    "UU":             "float— up→up latency",
    "UD":             "float— up(prev)→down(next): flight time (can be negative)",
    "DU":             "float— down(prev)→up(next): hold overlap",

    "total_elapsed_s":   "float— wall time from first key to Enter (seconds)",
    "chars_per_second":  "float— typing speed",
    "chars_per_minute":  "float— typing speed (CPM)",
    "backspace_count":   "int  — correction events",
    "error_rate":        "float— backspaces / password_length",
    "overlap_count":     "int  — keys pressed while previous still held",
    "overlap_ratio":     "float— overlaps / bigrams",
    "rhythm_cv":         "float— coefficient of variation on UD flight times",
    "dwell_stats":       "dict — {mean, std, min, max, median} of dwell_times",
    "flight_stats":      "dict — {mean, std, min, max, median} of UD times",
    "total_dwell_time":  "float— sum of all dwell times",
    "total_flight_time": "float— sum of all UD times",
    "hold_ratio":        "float— total_dwell / total_elapsed",

    "dwell_vector":   "list[float] — dwell_time per key position",
    "UD_vector":      "list[float] — UD flight time per bigram",
    "DD_vector":      "list[float] — DD time per bigram",
    "UU_vector":      "list[float] — UU time per bigram",
    "DU_vector":      "list[float] — DU time per bigram",
    "vector_length":  "int         — number of keys in sequence",
}


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_feature_array(record: dict,
                           include_dwell: bool = True,
                           include_UD: bool = True,
                           include_DD: bool = False,
                           include_UU: bool = False,
                           include_DU: bool = False,
                           include_agg: bool = True) -> np.ndarray:
    
    fv     = record["feature_vectors"]
    n      = fv["vector_length"]
    n_bi   = n - 1

    cols = []

    if include_dwell:
        cols.append(np.array(fv["dwell_vector"],  dtype=np.float32).reshape(-1, 1))

    def bigram_col(key):
        v = np.zeros(n, dtype=np.float32)
        if n_bi > 0:
            v[1:] = fv[key][:n_bi]  
        return v.reshape(-1, 1)

    if include_UD: cols.append(bigram_col("UD_vector"))
    if include_DD: cols.append(bigram_col("DD_vector"))
    if include_UU: cols.append(bigram_col("UU_vector"))
    if include_DU: cols.append(bigram_col("DU_vector"))

    if not cols:
        raise ValueError("At least one feature type must be enabled.")

    per_step = np.concatenate(cols, axis=1)  
    return per_step


def extract_aggregate_features(record: dict) -> np.ndarray:
    sf = record["sequence_features"]

    def s(key, sub=None):
        v = sf.get(key)
        if v is None:
            return 0.0
        if sub:
            return v.get(sub) or 0.0
        return float(v)

    return np.array([
        s("chars_per_minute"),
        s("error_rate"),
        s("overlap_ratio"),
        s("rhythm_cv"),
        s("dwell_stats", "mean"),
        s("dwell_stats", "std"),
        s("flight_stats", "mean"),
        s("flight_stats", "std"),
        s("hold_ratio"),
        s("total_elapsed_s"),
    ], dtype=np.float32)


if TORCH_AVAILABLE:
    class KeystrokeDataset(Dataset):

        def __init__(self, jsonl_path: str,
                     include_DD: bool = False,
                     include_UU: bool = False,
                     include_DU: bool = False,
                     normalize:  bool = True):
            self.records    = load_jsonl(jsonl_path)
            self.include_DD = include_DD
            self.include_UU = include_UU
            self.include_DU = include_DU

            labels = sorted({r["metadata"]["user_label"] for r in self.records})
            self.label_to_int = {l: i for i, l in enumerate(labels)}
            self.int_to_label = {i: l for l, i in self.label_to_int.items()}
            self.n_classes    = len(labels)

            self.by_user: dict[str, list[int]] = defaultdict(list)
            for idx, r in enumerate(self.records):
                self.by_user[r["metadata"]["user_label"]].append(idx)

            self.normalize = normalize
            if normalize:
                self._compute_norm_stats()

        def _compute_norm_stats(self):
            all_feats = []
            for r in self.records:
                arr = extract_feature_array(
                    r,
                    include_DD=self.include_DD,
                    include_UU=self.include_UU,
                    include_DU=self.include_DU,
                )
                all_feats.append(arr)
            cat = np.concatenate(all_feats, axis=0)
            self.feat_mean = cat.mean(axis=0)
            self.feat_std  = cat.std(axis=0) + 1e-8  

        def _get_sequence(self, record: dict) -> torch.FloatTensor:
            arr = extract_feature_array(
                record,
                include_DD=self.include_DD,
                include_UU=self.include_UU,
                include_DU=self.include_DU,
            )
            if self.normalize:
                arr = (arr - self.feat_mean) / self.feat_std
            return torch.from_numpy(arr)

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx: int) -> dict:
            r = self.records[idx]
            return {
                "sequence":  self._get_sequence(r),
                "agg":       torch.from_numpy(extract_aggregate_features(r)),
                "label":     r["metadata"]["user_label"],
                "label_int": self.label_to_int[r["metadata"]["user_label"]],
                "pw_hash":   r["password_hash"],
                "length":    r["feature_vectors"]["vector_length"],
            }

    class TripletSampler(Dataset):

        def __init__(self, base_dataset: "KeystrokeDataset",
                     n_triplets: int = 50_000):
            self.ds         = base_dataset
            self.n_triplets = n_triplets
            self.users      = list(base_dataset.by_user.keys())
            assert len(self.users) >= 2, "Need at least 2 users for triplet sampling."

        def __len__(self):
            return self.n_triplets

        def __getitem__(self, _):
            user_a = random.choice(self.users)
            other_users = [u for u in self.users if u != user_a]
            user_b = random.choice(other_users)

            a_indices = self.ds.by_user[user_a]
            b_indices = self.ds.by_user[user_b]

            anchor_idx, pos_idx = random.sample(a_indices, min(2, len(a_indices)))
            neg_idx = random.choice(b_indices)

            def pack(item):
                return {k: item[k] for k in ("sequence", "agg", "label_int", "length")}

            return {
                "anchor":   pack(self.ds[anchor_idx]),
                "positive": pack(self.ds[pos_idx]),
                "negative": pack(self.ds[neg_idx]),
            }

    def pad_collate_fn(batch: list[dict]) -> dict:
        lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
        max_len = lengths.max().item()
        feat_dim = batch[0]["sequence"].shape[1]

        sequences = torch.zeros(len(batch), max_len, feat_dim)
        masks     = torch.zeros(len(batch), max_len, dtype=torch.bool)

        for i, item in enumerate(batch):
            T = item["length"]
            sequences[i, :T, :] = item["sequence"]
            masks[i, :T]        = True

        return {
            "sequences":    sequences,   
            "masks":        masks,        
            "lengths":      lengths,      
            "agg":          torch.stack([item["agg"]          for item in batch]),
            "label_ints":   torch.tensor([item["label_int"]   for item in batch]),
            "labels":       [item["label"]                     for item in batch],
        }


    def triplet_collate_fn(batch: list[dict]) -> dict:
        """Collate for TripletSampler batches."""
        def collate_part(key):
            part = [item[key] for item in batch]
            return pad_collate_fn(part)

        return {
            "anchor":   collate_part("anchor"),
            "positive": collate_part("positive"),
            "negative": collate_part("negative"),
        }

def demo_load(jsonl_path: str):
    records = load_jsonl(jsonl_path)
    if not records:
        print("No records found.")
        return

    r = records[0]
    print("\n── First record summary ──────────────────────────────────")
    print(f"  user           : {r['metadata']['user_label']}")
    print(f"  session_id     : {r['metadata']['session_id']}")
    print(f"  password_hash  : {r['password_hash'][:16]}…")
    print(f"  password_length: {r['password_length']}")
    print(f"  key events     : {len(r['keystroke_events'])}")
    print(f"  bigrams        : {len(r['bigrams'])}")

    sf = r["sequence_features"]
    print(f"  elapsed        : {sf['total_elapsed_s']*1000:.0f}ms")
    print(f"  CPM            : {sf['chars_per_minute']:.1f}")
    print(f"  dwell mean     : {sf['dwell_stats']['mean']*1000:.2f}ms")
    print(f"  flight (UD) mean: {sf['flight_stats']['mean']*1000:.2f}ms")

    fv = r["feature_vectors"]
    print(f"  dwell_vector   : {fv['dwell_vector']}")
    print(f"  UD_vector      : {fv['UD_vector']}")

    if TORCH_AVAILABLE:
        from keystroke_dataset import KeystrokeDataset
        ds  = KeystrokeDataset(jsonl_path)
        item = ds[0]
        print(f"\n── Dataset item tensor shapes ───────────────────────────")
        print(f"  sequence : {item['sequence'].shape}")
        print(f"  agg      : {item['agg'].shape}")
        print(f"  label    : {item['label']}  (int={item['label_int']})")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "keystroke_dataset.jsonl"
    demo_load(path)