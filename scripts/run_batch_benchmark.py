"""
Batch Benchmark Runner - Run benchmarks on all instances in a folder.

Features:
- **Parallel instance execution** - Multiple instances run concurrently
- Skip completed instances (resume capability)
- Individual error isolation (one failure doesn't stop others)
- Progress tracking with timestamps
- Master log file for overall progress
- Warning detection for suspicious results

Usage:
  # Run all instances in parallel (default 3 concurrent instances)
  uv run python scripts/run_batch_benchmark.py --base-dir benchmark/real_small_batch/lead_time_0 --model "google/gemini-3-pro-preview"

  # More parallelism (5 concurrent instances)
  uv run python scripts/run_batch_benchmark.py --base-dir benchmark/real_small_batch/lead_time_0 --model "google/gemini-3-pro-preview" --parallel-instances 5

  # Resume from previous run (skip completed)
  uv run python scripts/run_batch_benchmark.py --base-dir ... --skip-completed
"""

import os
import sys
import json
import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Fix for nested ProcessPoolExecutor issues - use 'spawn' instead of 'fork'
# This prevents "Broken pipe" errors when ProcessPoolExecutor is used within ProcessPoolExecutor
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark_all_strategies import (
    benchmark_all, 
    detect_promised_lead_time, 
    is_instance_completed,
    DEFAULT_MODEL,
    SCRIPTS
)


def find_instances(base_dir: str) -> list:
    """
    Recursively find all instance directories (directories containing test.csv and train.csv).
    
    Supports both flat structure (real data):
        base_dir/{article_id}/test.csv
    
    And nested structure (synthetic data):
        base_dir/{pattern}/{variant}/{realization}/test.csv
    """
    instances = []
    base_path = Path(base_dir)
    
    def search_recursive(path: Path):
        if not path.is_dir():
            return
        
        test_csv = path / "test.csv"
        train_csv = path / "train.csv"
        
        # If this directory has both test.csv and train.csv, it's an instance
        if test_csv.exists() and train_csv.exists():
            instances.append(str(path))
            return  # Don't search deeper
        
        # Otherwise, search subdirectories
        for item in sorted(path.iterdir()):
            if item.is_dir():
                search_recursive(item)
    
    search_recursive(base_path)
    return instances


def run_single_instance(args_tuple):
    """
    Run benchmark for a single instance. 
    This function is designed for parallel execution via ProcessPoolExecutor.
    """
    instance_dir, model_name, max_workers_per_instance, max_periods = args_tuple
    instance_name = os.path.basename(instance_dir)
    
    try:
        # Auto-detect promised lead time
        promised_lead_time = detect_promised_lead_time(instance_dir)
        if promised_lead_time is None:
            promised_lead_time = 0
        
        # Run benchmark
        benchmark_all(
            promised_lead_time=promised_lead_time,
            instance_dir=instance_dir,
            max_periods=max_periods,
            max_workers=max_workers_per_instance,
            model_name=model_name
        )
        
        # Check for warnings in results
        results_file = os.path.join(instance_dir, "benchmark_results.json")
        warnings = []
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                summary = data.get('summary', {})
                if summary.get('warnings'):
                    warnings = list(summary['warnings'].values())
            except Exception:
                pass
        
        # Verify completion
        if is_instance_completed(instance_dir):
            if warnings:
                return instance_name, "SUCCESS_WITH_WARNINGS", warnings
            return instance_name, "SUCCESS", None
        else:
            return instance_name, "INCOMPLETE", "Results incomplete after benchmark"
            
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return instance_name, "FAILED", error_msg


def run_batch(
    base_dir: str,
    model_name: str = None,
    max_workers_per_instance: int = 3,
    parallel_instances: int = 3,
    max_periods: int = None,
    skip_completed: bool = False,
    force: bool = False
):
    """Run benchmarks on all instances in a directory with parallel execution."""
    
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    base_dir = os.path.abspath(base_dir)
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return
    
    # Find all instances
    all_instances = find_instances(base_dir)
    if not all_instances:
        print(f"No instances found in {base_dir}")
        print("Expected subdirectories with test.csv and train.csv")
        return
    
    # Filter instances based on skip_completed
    instances_to_run = []
    already_completed = []
    for inst in all_instances:
        if not force and skip_completed and is_instance_completed(inst):
            already_completed.append(os.path.basename(inst))
        else:
            instances_to_run.append(inst)
    
    print("=" * 80)
    print("BATCH BENCHMARK RUNNER (PARALLEL MODE)")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Model: {model_name}")
    print(f"Total instances found: {len(all_instances)}")
    print(f"Already completed (skipped): {len(already_completed)}")
    print(f"Instances to run: {len(instances_to_run)}")
    print(f"Parallel instances: {parallel_instances}")
    print(f"Workers per instance: {max_workers_per_instance}")
    print(f"Skip completed: {skip_completed}")
    print(f"Force rerun: {force}")
    print("=" * 80)
    
    if not instances_to_run:
        print("\nAll instances already completed! Use --force to rerun.")
        return
    
    # Create master log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log_path = os.path.join(base_dir, f"batch_log_{timestamp}.txt")
    
    def log(msg):
        """Log to both console and master log file."""
        timestamped_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(timestamped_msg)
        try:
            with open(master_log_path, 'a', encoding='utf-8') as f:
                f.write(timestamped_msg + "\n")
        except Exception:
            pass
    
    log(f"Starting parallel batch run")
    log(f"Instances to process: {len(instances_to_run)}")
    log(f"Master log: {master_log_path}")
    
    # Track results
    completed = []
    completed_with_warnings = []
    skipped = already_completed.copy()
    failed = []
    
    # Prepare task arguments
    task_args = [
        (inst, model_name, max_workers_per_instance, max_periods)
        for inst in instances_to_run
    ]
    
    # Run instances in parallel
    log(f"\nStarting {len(task_args)} instances with {parallel_instances} parallel workers...")
    
    with ProcessPoolExecutor(max_workers=parallel_instances) as executor:
        future_to_instance = {
            executor.submit(run_single_instance, args): args[0]
            for args in task_args
        }
        
        completed_count = 0
        total_count = len(task_args)
        
        for future in as_completed(future_to_instance):
            instance_dir = future_to_instance[future]
            instance_name = os.path.basename(instance_dir)
            completed_count += 1
            
            try:
                name, status, detail = future.result()
                
                if status == "SUCCESS":
                    log(f"[{completed_count}/{total_count}] SUCCESS: {name}")
                    completed.append(name)
                elif status == "SUCCESS_WITH_WARNINGS":
                    log(f"[{completed_count}/{total_count}] SUCCESS (with warnings): {name}")
                    for warn in detail:
                        log(f"  ⚠️  {warn}")
                    completed_with_warnings.append((name, detail))
                elif status == "INCOMPLETE":
                    log(f"[{completed_count}/{total_count}] INCOMPLETE: {name} - {detail}")
                    failed.append((name, detail))
                else:  # FAILED
                    log(f"[{completed_count}/{total_count}] FAILED: {name}")
                    log(f"  Error: {detail[:200]}...")  # Truncate long errors
                    failed.append((name, detail))
                    
            except Exception as e:
                log(f"[{completed_count}/{total_count}] EXCEPTION: {instance_name} - {str(e)}")
                failed.append((instance_name, str(e)))
    
    # Summary
    log(f"\n{'='*80}")
    log("BATCH RUN SUMMARY")
    log(f"{'='*80}")
    log(f"Total instances: {len(all_instances)}")
    log(f"Completed successfully: {len(completed)}")
    log(f"Completed with warnings: {len(completed_with_warnings)}")
    log(f"Skipped (already done): {len(skipped)}")
    log(f"Failed: {len(failed)}")
    
    if completed_with_warnings:
        log(f"\nInstances with warnings:")
        for name, warnings in completed_with_warnings:
            log(f"  - {name}:")
            for w in warnings:
                log(f"      ⚠️  {w}")
    
    if failed:
        log(f"\nFailed instances:")
        for name, error in failed:
            log(f"  - {name}: {error[:100]}...")
    
    log(f"\nMaster log saved to: {master_log_path}")
    
    # Save summary to JSON
    summary_path = os.path.join(base_dir, f"batch_summary_{timestamp}.json")
    summary = {
        'timestamp': timestamp,
        'base_dir': base_dir,
        'model': model_name,
        'parallel_instances': parallel_instances,
        'workers_per_instance': max_workers_per_instance,
        'total_instances': len(all_instances),
        'completed': completed,
        'completed_with_warnings': [
            {'instance': name, 'warnings': warnings}
            for name, warnings in completed_with_warnings
        ],
        'skipped': skipped,
        'failed': [{'instance': name, 'error': str(error)[:500]} for name, error in failed]
    }
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        log(f"Summary saved to: {summary_path}")
    except Exception as e:
        log(f"Warning: Could not save summary: {e}")
    
    log(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run benchmarks on all instances in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parallelism (3 concurrent instances)
  python run_batch_benchmark.py --base-dir benchmark/real_small_batch/lead_time_0 --model "google/gemini-3-pro-preview"
  
  # More parallelism (5 concurrent instances, 5 workers each)
  python run_batch_benchmark.py --base-dir ... --parallel-instances 5 --max-workers 5
  
  # Resume (skip completed instances)
  python run_batch_benchmark.py --base-dir ... --skip-completed
  
  # Force rerun all
  python run_batch_benchmark.py --base-dir ... --force
        """
    )
    parser.add_argument('--base-dir', type=str, required=True,
                       help='Base directory containing instance subfolders')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       help=f'OpenRouter model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--parallel-instances', type=int, default=3,
                       help='Number of instances to run in parallel (default: 3)')
    parser.add_argument('--max-workers', type=int, default=3,
                       help='Max parallel workers per instance (default: 3)')
    parser.add_argument('--max-periods', type=int, default=None,
                       help='Max periods per test (default: all)')
    parser.add_argument('--skip-completed', action='store_true',
                       help='Skip instances that already have complete results')
    parser.add_argument('--force', action='store_true',
                       help='Force rerun all instances (overrides --skip-completed)')
    
    args = parser.parse_args()
    
    run_batch(
        base_dir=args.base_dir,
        model_name=args.model,
        max_workers_per_instance=args.max_workers,
        parallel_instances=args.parallel_instances,
        max_periods=args.max_periods,
        skip_completed=args.skip_completed,
        force=args.force
    )
