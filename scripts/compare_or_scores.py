#!/usr/bin/env python3
"""Compare OR scores from benchmark_results.json vs my evaluation"""

import json
from pathlib import Path
import numpy as np

results_dir = Path("results/gemini-3-flash_bench")

or_scores = []
batch_scores = {}

for trajectory_type in ['real_trajectory', 'synthetic_trajectory']:
    for lead_time in ['lead_time_0', 'lead_time_4', 'lead_time_stochastic']:
        batch_dir = results_dir / trajectory_type / lead_time
        if not batch_dir.exists():
            continue
        
        batch_name = f"{trajectory_type}/{lead_time}"
        batch_or_scores = []
        
        for json_file in batch_dir.rglob('benchmark_results.json'):
            with open(json_file) as f:
                data = json.load(f)
            
            if 'or' in data['results']:
                ratio = data['results']['or']['ratio_to_perfect']
                batch_or_scores.append(ratio)
                or_scores.append(ratio)
        
        if batch_or_scores:
            batch_scores[batch_name] = {
                'score': float(np.mean(batch_or_scores)),
                'num_instances': len(batch_or_scores)
            }

print("=" * 70)
print("OR Agent Scores from benchmark_results.json")
print("=" * 70)
print(f"\nOverall Score: {np.mean(or_scores):.4f}")
print(f"Total Instances: {len(or_scores)}")
print(f"\nBatch Scores:")
for batch_name in sorted(batch_scores.keys()):
    stats = batch_scores[batch_name]
    print(f"  {batch_name:40s}: {stats['score']:.4f} ({stats['num_instances']} instances)")

print(f"\n{'=' * 70}")
print("Comparison with evaluate_results.py output:")
print("=" * 70)

eval_file = Path("results/gemini-3-flash_bench_or_evaluation.json")
if eval_file.exists():
    with open(eval_file) as f:
        eval_data = json.load(f)
    
    print(f"\nEvaluated Overall Score: {eval_data['overall_score']:.4f}")
    print(f"\nDifference: {abs(eval_data['overall_score'] - np.mean(or_scores)):.4f}")
    
    print(f"\nBatch-wise comparison:")
    for batch_name in sorted(batch_scores.keys()):
        orig = batch_scores[batch_name]['score']
        eval_score = eval_data['batches'][batch_name]['score']
        diff = abs(orig - eval_score)
        print(f"  {batch_name:40s}: orig={orig:.4f}, eval={eval_score:.4f}, diff={diff:.4f}")
