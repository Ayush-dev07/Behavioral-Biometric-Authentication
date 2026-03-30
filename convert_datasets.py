import sys
import json
import csv
import hashlib
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

class CMUConverter:
    PASSWORD = ".tie5Roanl"
    
    def __init__(self):
        self.password_hash = hashlib.sha256(self.PASSWORD.encode()).hexdigest()
    
    def convert(self, input_path: str) -> List[Dict]:
        records = []
        
        with open(input_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                user_label = row['subject']
                session_index = int(row['sessionIndex'])
                rep_index = int(row['rep'])
                
                dwell_times = []
                dd_times = []
                
                for char in ['.', 't', 'i', 'e', '5', 'Shift.r', 'o', 'a', 'n', 'l']:
                    col_name = f'H.{char}'
                    if col_name in row:
                        dwell_times.append(float(row[col_name]))
                
                for i, char in enumerate(['.', 't', 'i', 'e', '5', 'Shift.r', 'o', 'a', 'n', 'l']):
                    if i == 0:
                        continue
                    col_name = f'DD.{char}'
                    if col_name in row:
                        dd_times.append(float(row[col_name]))
                
                ud_times = []
                for i in range(len(dd_times)):
                    if i < len(dwell_times):
                        ud = dd_times[i] - dwell_times[i]
                        ud_times.append(ud)
                
                keystroke_events = []
                cumulative_time = 0.0
                
                for i, (char, dwell) in enumerate(zip(self.PASSWORD, dwell_times)):
                    press_time = cumulative_time
                    release_time = press_time + dwell
                    
                    keystroke_events.append({
                        'key_id': f'char:{char}',
                        'position': i,
                        'press_ts': press_time,
                        'release_ts': release_time,
                        'dwell_time': dwell,
                        'is_special': False,
                        'is_modifier': False,
                    })
                    
                    if i < len(dd_times):
                        cumulative_time += dd_times[i]
                    else:
                        cumulative_time = release_time
                
                bigrams = []
                for i in range(len(keystroke_events) - 1):
                    a = keystroke_events[i]
                    b = keystroke_events[i + 1]
                    
                    bigrams.append({
                        'from_key': a['key_id'],
                        'to_key': b['key_id'],
                        'from_pos': a['position'],
                        'to_pos': b['position'],
                        'DD': dd_times[i] if i < len(dd_times) else 0,
                        'UD': ud_times[i] if i < len(ud_times) else 0,
                        'UU': b['release_ts'] - a['release_ts'],
                        'DU': b['release_ts'] - a['press_ts'],
                    })
                
                total_elapsed = keystroke_events[-1]['release_ts'] - keystroke_events[0]['press_ts']
                cpm = (len(self.PASSWORD) / total_elapsed) * 60 if total_elapsed > 0 else 0
                
                sequence_features = {
                    'total_elapsed_s': total_elapsed,
                    'chars_per_minute': cpm,
                    'error_rate': 0,  
                    'backspace_count': 0,
                    'rhythm_cv': float(np.std(ud_times) / (np.mean(ud_times) + 1e-8)) if ud_times else 0,
                }
                
                feature_vectors = {
                    'dwell_vector': dwell_times,
                    'UD_vector': ud_times,
                    'DD_vector': dd_times,
                    'UU_vector': [b['UU'] for b in bigrams],
                    'DU_vector': [b['DU'] for b in bigrams],
                    'vector_length': len(dwell_times),
                }
                
                record = {
                    'metadata': {
                        'user_label': user_label,
                        'session_id': f'cmu_{user_label}_session{session_index}',
                        'sample_index': rep_index,
                        'collection_ts': None,
                        'platform': 'CMU Benchmark Dataset',
                    },
                    'keystroke_events': keystroke_events,
                    'bigrams': bigrams,
                    'sequence_features': sequence_features,
                    'feature_vectors': feature_vectors,
                    'password_correct': True,
                    'password_hash': self.password_hash,
                    'password_length': len(self.PASSWORD),
                }
                
                records.append(record)
        
        return records

class AaltoConverter:
    def convert(self, input_path: str) -> List[Dict]:
        sessions = defaultdict(lambda: defaultdict(list))
        
        with open(input_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                user_id = row['USER_ID']
                session_id = row['SESSION_ID']
                keycode = int(row['KEYCODE'])
                press_time = float(row['PRESS_TIME'])
                release_time = float(row['RELEASE_TIME'])
                
                sessions[user_id][session_id].append({
                    'keycode': keycode,
                    'char': self._keycode_to_char(keycode),
                    'press_time': press_time,
                    'release_time': release_time,
                })
        
        records = []
        
        for user_id, user_sessions in sessions.items():
            for session_id, keystrokes in user_sessions.items():
    
                keystrokes.sort(key=lambda k: k['press_time'])
                
    
                password = ''.join([k['char'] for k in keystrokes if k['char']])
                
                if len(password) < 5:
                    continue  
                
                record = self._create_record(user_id, session_id, keystrokes, password)
                records.append(record)
        
        return records
    
    def _keycode_to_char(self, keycode: int) -> str:

        if 65 <= keycode <= 90:  # A-Z
            return chr(keycode).lower()
        elif 97 <= keycode <= 122:  # a-z
            return chr(keycode)
        elif 48 <= keycode <= 57:  # 0-9
            return chr(keycode)
        else:
            return chr(keycode) if 32 <= keycode <= 126 else ''
    
    def _create_record(self, user_id, session_id, keystrokes, password):
        start_time = keystrokes[0]['press_time']
        
        keystroke_events = []
        for i, k in enumerate(keystrokes):
            if not k['char']:
                continue
            
            keystroke_events.append({
                'key_id': f"char:{k['char']}",
                'position': len(keystroke_events),
                'press_ts': k['press_time'] - start_time,
                'release_ts': k['release_time'] - start_time,
                'dwell_time': k['release_time'] - k['press_time'],
                'is_special': False,
                'is_modifier': False,
            })
        
        bigrams = []
        for i in range(len(keystroke_events) - 1):
            a, b = keystroke_events[i], keystroke_events[i + 1]
            bigrams.append({
                'from_key': a['key_id'],
                'to_key': b['key_id'],
                'from_pos': a['position'],
                'to_pos': b['position'],
                'DD': b['press_ts'] - a['press_ts'],
                'UU': b['release_ts'] - a['release_ts'],
                'UD': b['press_ts'] - a['release_ts'],
                'DU': b['release_ts'] - a['press_ts'],
            })
        
        dwell_times = [e['dwell_time'] for e in keystroke_events]
        ud_times = [b['UD'] for b in bigrams]
        
        total_elapsed = keystroke_events[-1]['release_ts'] - keystroke_events[0]['press_ts']
        
        return {
            'metadata': {
                'user_label': f'aalto_user{user_id}',
                'session_id': session_id,
                'sample_index': 0,
                'platform': 'Aalto Dataset',
            },
            'keystroke_events': keystroke_events,
            'bigrams': bigrams,
            'sequence_features': {
                'total_elapsed_s': total_elapsed,
                'chars_per_minute': (len(password) / total_elapsed) * 60,
                'rhythm_cv': float(np.std(ud_times) / (np.mean(ud_times) + 1e-8)),
            },
            'feature_vectors': {
                'dwell_vector': dwell_times,
                'UD_vector': ud_times,
                'DD_vector': [b['DD'] for b in bigrams],
                'UU_vector': [b['UU'] for b in bigrams],
                'DU_vector': [b['DU'] for b in bigrams],
                'vector_length': len(dwell_times),
            },
            'password_correct': True,
            'password_hash': hashlib.sha256(password.encode()).hexdigest(),
            'password_length': len(password),
        }

def main():
    parser = argparse.ArgumentParser(
        description="Convert public keystroke datasets to training format"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['cmu', 'aalto', 'buffalo', 'greyc'],
        help="Dataset type to convert"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Input file path"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='keystroke_dataset.jsonl',
        help="Output JSONL file path"
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=10,
        help="Minimum samples per user to include"
    )
    
    args = parser.parse_args()
    
    converters = {
        'cmu': CMUConverter(),
        'aalto': AaltoConverter(),
    }
    
    if args.dataset not in converters:
        print(f"[ERROR] Converter for {args.dataset} not yet implemented.")
        print("        Available: cmu, aalto")
        sys.exit(1)
    
    converter = converters[args.dataset]
    
    # Convert
    print(f"\nConverting {args.dataset} dataset...")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}\n")
    
    records = converter.convert(args.input)
    
    user_counts = defaultdict(int)
    for r in records:
        user_counts[r['metadata']['user_label']] += 1
    
    valid_users = {u for u, c in user_counts.items() if c >= args.min_samples}
    filtered_records = [r for r in records if r['metadata']['user_label'] in valid_users]
    
    print(f"Total records:       {len(records)}")
    print(f"Filtered records:    {len(filtered_records)}")
    print(f"Users (before):      {len(user_counts)}")
    print(f"Users (after):       {len(valid_users)}")
    print(f"Min samples/user:    {args.min_samples}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in filtered_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"\n✅ Conversion complete!")
    print(f"   Output: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    print(f"\nSamples per user:")
    user_sample_counts = defaultdict(int)
    for r in filtered_records:
        user_sample_counts[r['metadata']['user_label']] += 1
    
    counts = sorted(user_sample_counts.values())
    print(f"  Min:    {min(counts)}")
    print(f"  Max:    {max(counts)}")
    print(f"  Mean:   {np.mean(counts):.1f}")
    print(f"  Median: {np.median(counts):.0f}")


if __name__ == "__main__":
    main()