"""
Benchmark script to compare all strategies.

Runs each strategy and calculates:
- Average reward
- Standard deviation
- Min/Max rewards

Deterministic strategies (or, perfect_score) run once.
LLM-based strategies (llm, llm_to_or, or_to_llm) run once.

Usage:
  uv run python scripts/benchmark_all_strategies.py \
    --directory benchmark/synthetic_trajectory/lead_time_0/p01_stationary_iid/v1_normal_100_25/r1_med

  # Model can be specified (default: x-ai/grok-4.1-fast)
  uv run python scripts/benchmark_all_strategies.py --directory ... --model google/gemini-3-flash-preview

  # Promised lead time auto-detected from folder path:
  #   lead_time_0 → promised_lead_time=0
  #   lead_time_4 → promised_lead_time=4
  #   lead_time_stochastic → promised_lead_time=2
"""

import os
import sys
import subprocess
import re
import json
import argparse
import multiprocessing
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple

# Get the current Python executable
PYTHON_EXECUTABLE = sys.executable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Scripts to benchmark (5 strategies)
SCRIPTS = {
    "or": "scripts/run_or.py",
    "llm": "scripts/run_llm.py",
    "llm_to_or": "scripts/run_llm_to_or.py",
    "or_to_llm": "scripts/run_or_to_llm.py",
    "perfect_score": "scripts/perfect_score.py",
}

# Deterministic scripts (run only once, no LLM involved)
DETERMINISTIC_SCRIPTS = {"or", "perfect_score"}

# LLM-based scripts (run once per instance, averaging across many instances)
LLM_SCRIPTS = {"llm", "llm_to_or", "or_to_llm"}

# Number of runs for LLM-based scripts
NUM_RUNS = 1

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Default model for OpenRouter
# NOTE: this is the model used by run_batch_benchmark.py unless overridden via --model
DEFAULT_MODEL = "x-ai/grok-4.1-fast"


def is_instance_completed(instance_dir: str, required_strategies: set = None) -> bool:
    """
    Check if an instance already has complete benchmark results.
    
    Returns True if benchmark_results.json exists and contains results for all required strategies.
    """
    if required_strategies is None:
        required_strategies = set(SCRIPTS.keys())
    
    results_file = os.path.join(instance_dir, "benchmark_results.json")
    if not os.path.exists(results_file):
        return False
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', {})
        for strategy in required_strategies:
            if strategy not in results:
                return False
            if not results[strategy].get('rewards') or len(results[strategy]['rewards']) == 0:
                return False
        
        return True
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def detect_promised_lead_time(instance_dir: str) -> int:
    """
    Auto-detect promised lead time from folder path.
    
    Looks for lead_time_0, lead_time_4, or lead_time_stochastic in path.
    Returns:
        - 0 if 'lead_time_0' found in path
        - 4 if 'lead_time_4' found in path
        - 2 if 'lead_time_stochastic' found in path
        - None if no pattern found
    """
    path_str = str(instance_dir).replace('\\', '/')
    
    if 'lead_time_0/' in path_str or '/lead_time_0' in path_str:
        return 0
    elif 'lead_time_4/' in path_str or '/lead_time_4' in path_str:
        return 4
    elif 'lead_time_stochastic/' in path_str or '/lead_time_stochastic' in path_str:
        return 2
    
    return None


def extract_reward_from_output(output: str) -> tuple:
    """
    Extract total reward from script output.
    Returns (reward, error_msg) where error_msg is None if no issues detected.
    """
    # First check for VM Final Reward to detect crashes
    vm_final_match = re.search(r"VM Final Reward:\s*(-?[\d,]+\.?\d*)", output)
    vm_final_reward = None
    if vm_final_match:
        try:
            vm_final_reward = float(vm_final_match.group(1).replace(',', '').replace(' ', ''))
        except ValueError:
            pass
    
    # Pattern to match: >>> Total Reward: $1234.56 <<< or $-123.45 <<<
    # or variations like "Total Reward (OR Baseline): $1234.56"
    # or Perfect Score: $1234.56
    # Note: Must handle negative numbers correctly!
    patterns = [
        r">>>\s*Perfect Score:\s*\$(-?\s*[\d,]+\.?\d*)\s*<<<",  # Perfect Score format
        r">>>\s*Total Reward[^:]*:\s*\$(-?\s*[\d,]+\.?\d*)\s*<<<",  # With >>> ... <<< (supports negative)
        r"Total Reward[^:]*:\s*\$(-?\s*[\d,]+\.?\d*)",  # Without >>> ... <<< (supports negative)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            try:
                # Remove commas and whitespace from number string
                reward_str = match.group(1).replace(',', '').replace(' ', '')
                reward = float(reward_str)
                
                # Check for VM crash indicator (VM Final Reward = -1)
                if vm_final_reward is not None and vm_final_reward == -1.0:
                    return (reward, "VM_CRASHED: Final Reward = -1.0")
                
                # Check for suspicious $0.00 reward
                if reward == 0.0 and vm_final_reward == -1.0:
                    return (reward, "VM_CRASHED: Zero reward with Final Reward = -1.0")
                
                return (reward, None)
            except ValueError:
                continue
    
    # If no match found, try to find any number after "Total Reward"
    lines = output.split('\n')
    for line in lines:
        if 'Total Reward' in line or 'total_reward' in line.lower():
            # Try to extract number (handle comma-separated numbers and negative signs)
            # Look for pattern: $ followed by optional minus, then number
            numbers = re.findall(r'\$(-?\s*[\d,]+\.?\d*)', line)
            if numbers:
                try:
                    # Take the last number and remove commas and whitespace
                    reward_str = numbers[-1].replace(',', '').replace(' ', '')
                    reward = float(reward_str)
                    
                    # Check for VM crash
                    if vm_final_reward is not None and vm_final_reward == -1.0:
                        return (reward, "VM_CRASHED: Final Reward = -1.0")
                    
                    return (reward, None)
                except ValueError:
                    continue
    
    # Fallback to VM Final Reward if no other patterns match
    if vm_final_reward is not None and vm_final_reward != -1.0:
        return (vm_final_reward, "Used VM Final Reward as fallback")
    
    return (None, "PARSE_ERROR: Could not extract reward from output")


def run_script(script_path: str, run_num: int, script_name: str,
               promised_lead_time: int, instance_dir: str,
               max_periods: int = None, base_dir: str = None,
               model_name: str = None, perfect_reward: float = None) -> Tuple[float, str, str]:
    """
    Run a script for a given instance and return the reward and output.
    
    Returns:
        (reward, error_message, output)
    """
    # ... (same setup) ...
    if base_dir is None:
        base_dir = str(BASE_DIR)
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    # ... (same cmd construction) ...
    test_file = os.path.join(instance_dir, "test.csv")
    train_file = os.path.join(instance_dir, "train.csv")
    
    if not os.path.exists(test_file):
        return None, f"Test file not found: {test_file}", ""
    
    # perfect_score.py only needs --demand-file
    if script_name == "perfect_score":
        cmd = [
            PYTHON_EXECUTABLE, script_path,
            "--demand-file", test_file,
        ]
    elif script_name == "or":
        # OR doesn't need --model
        if not os.path.exists(train_file):
            return None, f"Train file not found: {train_file}", ""
        
        cmd = [
            PYTHON_EXECUTABLE, script_path,
            "--demand-file", test_file,
            "--promised-lead-time", str(promised_lead_time),
            "--real-instance-train", train_file,
        ]
        if max_periods is not None:
            cmd.extend(["--max-periods", str(max_periods)])
    else:
        # LLM scripts need --model parameter
        if not os.path.exists(train_file):
            return None, f"Train file not found: {train_file}", ""
        
        cmd = [
            PYTHON_EXECUTABLE, script_path,
            "--demand-file", test_file,
            "--promised-lead-time", str(promised_lead_time),
            "--real-instance-train", train_file,
            "--model", model_name,
        ]
        if max_periods is not None:
            cmd.extend(["--max-periods", str(max_periods)])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=base_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",  # Force UTF-8 so output with Unicode (e.g. product descriptions) decodes correctly on Windows (default gbk would fail)
            errors="replace",
            timeout=7200,  # 2 hour timeout (7200 seconds)
            stdin=subprocess.DEVNULL,  # Prevent waiting for stdin input
        )
        
        output = (result.stdout or "") + (result.stderr or "")
        
        # Add timestamp and perfect reward to output for logging
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output = f"[Timestamp: {timestamp}]\n[Script: {script_name}]\n[Instance: {instance_dir}]\n\n" + output
        
        if perfect_reward is not None:
            output += f"\n\n{'='*40}\nREFERENCE: Perfect Score: ${perfect_reward:.2f}\n{'='*40}\n"
        
        # Save output to log file
        log_filename = f"{script_name}_{run_num}.txt"
        log_path = os.path.join(instance_dir, log_filename)
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(output)
        except Exception as e:
            # If we can't write the log, continue anyway but note it
            output += f"\n[Warning: Could not save log to {log_path}: {str(e)}]"
        
        # Try to extract reward from output
        reward, extraction_error = extract_reward_from_output(output)
        
        # If extraction failed, try reading from log file (in case output was truncated)
        # This handles cases where the script completed but subprocess output was incomplete
        # Also handles encoding issues or buffering problems
        if reward is None and os.path.exists(log_path):
            try:
                # Try multiple encodings in case of encoding issues
                for encoding in ['utf-8', 'utf-16', 'latin-1']:
                    try:
                        with open(log_path, 'r', encoding=encoding, errors='replace') as f:
                            log_content = f.read()
                        reward, extraction_error = extract_reward_from_output(log_content)
                        if reward is not None:
                            break  # Successfully extracted, no need to try other encodings
                    except (UnicodeDecodeError, UnicodeError):
                        continue  # Try next encoding
            except Exception:
                pass
        
        # If we successfully extracted reward, return it even if exit code is non-zero
        # Script may have completed successfully but returned non-zero for other reasons
        # (e.g., API rate limit errors after completion, cleanup errors, etc.)
        if reward is not None:
            # Check for VM crash or other extraction errors
            if extraction_error and extraction_error.startswith("VM_CRASHED"):
                return reward, f"ERROR: {extraction_error}", output
            if result.returncode != 0:
                # Log a warning but still return the reward
                return reward, f"Warning: Script returned exit code {result.returncode} but reward was extracted", output
            if extraction_error:
                return reward, f"Warning: {extraction_error}", output
            return reward, None, output
        
        # If we couldn't extract reward, return error
        return None, f"Could not extract reward from output. Exit code: {result.returncode}. {extraction_error or ''}", output
        
    except subprocess.TimeoutExpired as e:
        timeout_minutes = 7200 / 60
        return None, f"Script timed out after {timeout_minutes:.1f} minutes ({7200} seconds)", ""
    except Exception as e:
        return None, f"Error running script: {str(e)}", ""


def run_single_task(task_info):
    """Wrapper function for parallel execution."""
    script_path, run_num, script_name, promised_lead_time, instance_dir, max_periods, base_dir, model_name, perfect_reward = task_info
    return (script_name, run_num), run_script(
        script_path, run_num, script_name, promised_lead_time, instance_dir, max_periods, base_dir, model_name, perfect_reward
    )


def benchmark_all(promised_lead_time: int, instance_dir: str, max_periods: int = None, 
                  max_workers: int = None, model_name: str = None):
    """Run all benchmarks and collect results."""
    
    # Validate instance directory
    instance_dir = os.path.abspath(instance_dir)
    if not os.path.exists(instance_dir):
        print(f"Error: Directory not found: {instance_dir}")
        return
    
    test_file = os.path.join(instance_dir, "test.csv")
    train_file = os.path.join(instance_dir, "train.csv")
    
    if not os.path.exists(test_file):
        print(f"Error: test.csv not found in {instance_dir}")
        return
    if not os.path.exists(train_file):
        print(f"Error: train.csv not found in {instance_dir}")
        return
    
    instance_name = os.path.basename(instance_dir)
    
    # Set default max_workers for LLM scripts
    # IMPORTANT: LLM API rate limits are the main bottleneck, not CPU/IO
    # Most LLM APIs have strict per-minute token limits:
    #   - Gemini: 1M tokens/min (shared across all concurrent calls)
    #   - OpenAI: varies by tier (500-10000 RPM)
    # Too many parallel workers will trigger 429 RESOURCE_EXHAUSTED errors
    if max_workers is None:
        # Conservative default: 5 workers to avoid API rate limits
        # Each worker runs ~50 LLM calls, so 5 workers = ~250 concurrent call capacity
        max_workers = 5
        print(f"Note: Using {max_workers} parallel workers (conservative default to avoid API rate limits). "
              f"Use --max-workers to increase if your API quota allows.")
    
    # Calculate total runs
    total_llm_runs = len(LLM_SCRIPTS) * NUM_RUNS
    total_deterministic_runs = len(DETERMINISTIC_SCRIPTS)
    total_runs = total_llm_runs + total_deterministic_runs
    
    print("=" * 80)
    print("BENCHMARK CONFIGURATION")
    print("=" * 80)
    print(f"Instance directory: {instance_dir}")
    print(f"Instance name: {instance_name}")
    print(f"Promised lead time: {promised_lead_time}")
    print(f"Model: {model_name}")
    print(f"Strategies: {', '.join(SCRIPTS.keys())}")
    print(f"  - Deterministic (1 run each): {', '.join(DETERMINISTIC_SCRIPTS)}")
    print(f"  - LLM-based ({NUM_RUNS} run each): {', '.join(LLM_SCRIPTS)}")
    print(f"Total runs: {total_runs}")
    if max_periods:
        print(f"Max periods: {max_periods}")
    else:
        print(f"Max periods: All (no limit)")
    print(f"Parallel workers: {max_workers}")
    print("=" * 80)
    
    # API rate limit warning
    # High parallelism can trigger 429 RESOURCE_EXHAUSTED errors from LLM APIs
    if max_workers >= 10:
        print("\n⚠️  WARNING: High API rate limit risk!")
        print(f"   - {total_llm_runs} LLM tasks will run with {max_workers} parallel workers")
        print(f"   - This may trigger API rate limits (429 RESOURCE_EXHAUSTED)")
        print(f"   - Gemini: 1M tokens/min limit, OpenAI: varies by tier")
        print(f"   - Consider reducing --max-workers if you see rate limit errors")
        print(f"   - Recommended: 3-5 workers for most API tiers\n")
    
    # Results structure: results[script_name] = [reward1, reward2, ...]
    results = defaultdict(list)
    errors = defaultdict(list)
    
    # Use default model if not specified
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    # 1. Run perfect_score first to get a baseline for all logs
    perfect_reward = None
    if "perfect_score" in SCRIPTS:
        print(f"\n[{'perfect_score'}] Running to get baseline...")
        script_path = BASE_DIR / SCRIPTS["perfect_score"]
        if script_path.exists():
            rew, err, out = run_script(
                str(script_path), 1, "perfect_score",
                promised_lead_time, instance_dir, max_periods, model_name=model_name
            )
            if not err:
                perfect_reward = rew
                results["perfect_score"].append(rew)
                print(f"[{'perfect_score'}] Baseline reward: ${rew:.2f}")
            else:
                print(f"[{'perfect_score'}] Error getting baseline: {err}")
                errors["perfect_score"].append(err)

    completed_runs = len(results["perfect_score"]) + len(errors["perfect_score"])
    
    # Prepare all LLM tasks upfront
    all_llm_tasks = []
    for script_name, script_path in SCRIPTS.items():
        if script_name in LLM_SCRIPTS:
            script_full_path = BASE_DIR / script_path
            if script_full_path.exists():
                for run_num in range(1, NUM_RUNS + 1):
                    all_llm_tasks.append((
                        str(script_full_path), run_num, script_name,
                        promised_lead_time, instance_dir, max_periods, str(BASE_DIR), model_name, perfect_reward
                    ))
    
    # Run all LLM scripts in parallel with batching to prevent system overload
    if all_llm_tasks:
        print(f"\nRunning {len(all_llm_tasks)} LLM tasks in parallel ({max_workers} workers)...")
        print(f"This includes: {', '.join(LLM_SCRIPTS)}")
        print(f"Note: Tasks will be executed in batches to prevent system overload.\n")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            task_index = 0
            active_futures = set()
            
            # Submit tasks in batches (don't submit all at once)
            while task_index < len(all_llm_tasks) or active_futures:
                # Submit new tasks up to max_workers limit
                while len(active_futures) < max_workers and task_index < len(all_llm_tasks):
                    task = all_llm_tasks[task_index]
                    future = executor.submit(run_single_task, task)
                    future_to_task[future] = task
                    active_futures.add(future)
                    task_index += 1
                
                # Wait for at least one task to complete
                if active_futures:
                    done, not_done = wait(active_futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        active_futures.remove(future)
                        completed_runs += 1
                        task = future_to_task[future]
                        script_name = task[2]
                        run_num = task[1]
                        
                        try:
                            _, (reward, error, output) = future.result()
                            
                            if error:
                                print(f"[{script_name}] Run {run_num}/{NUM_RUNS} ({completed_runs}/{len(all_llm_tasks)}): ERROR: {error}")
                                errors[script_name].append(error)
                            else:
                                perf_str = f" (Perfect: ${perfect_reward:.2f})" if perfect_reward is not None else ""
                                print(f"[{script_name}] Run {run_num}/{NUM_RUNS} ({completed_runs}/{len(all_llm_tasks)}): Reward: ${reward:.2f}{perf_str} (log saved)")
                                results[script_name].append(reward)
                        except Exception as e:
                            print(f"[{script_name}] Run {run_num}/{NUM_RUNS} ({completed_runs}/{len(all_llm_tasks)}): EXCEPTION: {str(e)}")
                            errors[script_name].append(f"Exception: {str(e)}")
    
    # Run deterministic scripts sequentially (fast, no need for parallel, only 1 run each)
    print(f"\n{'='*80}")
    print("Running deterministic scripts (1 run each)...")
    print(f"{'='*80}\n")
    
    for script_name in DETERMINISTIC_SCRIPTS:
        if script_name == "perfect_score":
            continue # Already ran first
            
        script_path = SCRIPTS[script_name]
        script_full_path = BASE_DIR / script_path
        
        if not script_full_path.exists():
            print(f"[{script_name}] ERROR: Script not found: {script_full_path}")
            continue
        
        completed_runs += 1
        print(f"[{script_name}] Running 1 run ({completed_runs}/{total_runs})...", end=" ", flush=True)
        
        reward, error, output = run_script(
            str(script_full_path), 1, script_name,
            promised_lead_time, instance_dir, max_periods, model_name=model_name,
            perfect_reward=perfect_reward
        )
        
        if error:
            print(f"ERROR: {error}")
            errors[script_name].append(error)
        else:
            perf_str = f" (Perfect: ${perfect_reward:.2f})" if perfect_reward is not None else ""
            print(f"Reward: ${reward:.2f}{perf_str} (log saved)")
            results[script_name].append(reward)
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nInstance: {instance_name}")
    print(f"Promised Lead Time: {promised_lead_time}")
    if perfect_reward is not None:
        print(f"Perfect Score: ${perfect_reward:.2f}")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Avg Reward':<15} {'Ratio':<10} {'Std Dev':<12} {'Min':<12} {'Max':<12} {'Runs':<6}")
    print("-" * 80)
    
    overall_stats = {}
    for script_name in SCRIPTS.keys():
        rewards = results[script_name]
        if rewards:
            mean_reward = np.mean(rewards)
            # Calculate ratio relative to perfect score
            ratio = mean_reward / perfect_reward if perfect_reward and perfect_reward > 0 else None
            overall_stats[script_name] = {
                'mean': mean_reward,
                'std': np.std(rewards) if len(rewards) > 1 else 0.0,
                'min': np.min(rewards),
                'max': np.max(rewards),
                'count': len(rewards),
                'ratio': ratio,
            }
            ratio_str = f"{ratio:.2%}" if ratio is not None else "N/A"
            print(f"{script_name:<20} ${mean_reward:<14.2f} {ratio_str:<10} "
                  f"${overall_stats[script_name]['std']:<11.2f} "
                  f"${overall_stats[script_name]['min']:<11.2f} "
                  f"${overall_stats[script_name]['max']:<11.2f} "
                  f"{overall_stats[script_name]['count']:<6}")
        else:
            print(f"{script_name:<20} {'N/A':<15} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'0':<6}")
    
    # Show errors if any
    print("\n" + "-" * 80)
    has_errors = False
    for script_name in SCRIPTS.keys():
        if errors[script_name]:
            has_errors = True
            print(f"\nErrors for {script_name}:")
            for error in errors[script_name]:
                print(f"  - {error}")
    
    if not has_errors:
        print("\nNo errors encountered.")
    
    # Print ratio summary
    if perfect_reward and perfect_reward > 0:
        print("\n" + "=" * 80)
        print("PERFORMANCE RATIO (relative to Perfect Score)")
        print("=" * 80)
        print(f"Perfect Score: ${perfect_reward:.2f} (100.00%)")
        print("-" * 40)
        for script_name in SCRIPTS.keys():
            if script_name == "perfect_score":
                continue
            if script_name in overall_stats and overall_stats[script_name].get('ratio') is not None:
                ratio = overall_stats[script_name]['ratio']
                mean = overall_stats[script_name]['mean']
                print(f"{script_name:<15} ${mean:>10.2f}  ({ratio:>6.2%})")
        print("=" * 80)
    
    # Save detailed results to JSON
    output_file = os.path.join(instance_dir, "benchmark_results.json")
    
    # Build results with ratio
    results_with_ratio = {}
    for script_name in SCRIPTS.keys():
        rewards = results[script_name]
        if rewards:
            mean_reward = float(np.mean(rewards))
            ratio = mean_reward / perfect_reward if perfect_reward and perfect_reward > 0 else None
            results_with_ratio[script_name] = {
                'rewards': [float(r) for r in rewards],
                'mean': mean_reward,
                'std': float(np.std(rewards)) if len(rewards) > 1 else 0.0,
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'count': len(rewards),
                'ratio_to_perfect': ratio,
            }
        else:
            results_with_ratio[script_name] = {
                'rewards': [],
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'count': 0,
                'ratio_to_perfect': None,
            }
    
    # Note: Negative rewards are normal (e.g., with lead time > 0), no warnings needed
    warnings = {}
    
    # Build summary
    summary = {
        'perfect_score': perfect_reward,
        'ratios': {
            script_name: results_with_ratio[script_name]['ratio_to_perfect']
            for script_name in SCRIPTS.keys()
            if results_with_ratio[script_name]['ratio_to_perfect'] is not None
        },
        'warnings': warnings if warnings else None
    }
    
    detailed_results = {
        'instance_dir': instance_dir,
        'instance_name': instance_name,
        'promised_lead_time': promised_lead_time,
        'model': model_name,
        'num_runs_llm': NUM_RUNS,
        'num_runs_deterministic': 1,
        'max_periods': max_periods,
        'summary': summary,
        'results': results_with_ratio,
        'errors': {
            script_name: errors[script_name]
            for script_name in SCRIPTS.keys()
            if errors[script_name]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n\nDetailed results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    # Fix for ProcessPoolExecutor issues - use 'spawn' instead of 'fork'
    # This prevents hanging and "Broken pipe" errors with multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(
        description='Benchmark all strategies (or, llm, llm_to_or, or_to_llm, perfect_score)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  - Deterministic (1 run): or, perfect_score
  - LLM-based (1 run): llm, llm_to_or, or_to_llm

Promised lead time auto-detection:
  - lead_time_0 folder → promised_lead_time=0
  - lead_time_4 folder → promised_lead_time=4
  - lead_time_stochastic folder → promised_lead_time=2

Example:
  python benchmark_all_strategies.py --directory D:\\OR_Agent\\benchmark\\small_batch_4_1\\lead_time_0\\p01_stationary_iid\\v1_normal_100_25\\r1
  python benchmark_all_strategies.py --directory ... --model google/gemini-3-flash-preview
        """
    )
    parser.add_argument('--promised-lead-time', type=int, default=None,
                       help='Promised lead time in periods. If not provided, auto-detected from folder path '
                            '(lead_time_0→0, lead_time_4→4, lead_time_stochastic→2)')
    parser.add_argument('--directory', type=str, required=True,
                       help='Path to instance directory containing test.csv and train.csv (required)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       help=f'OpenRouter model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--max-periods', type=int, default=None,
                       help='Maximum number of periods to run per test. Default: None (runs all periods)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers for LLM scripts. '
                             'Default: 5 (conservative to avoid API rate limits). '
                             'Increase cautiously based on your API quota.')
    args = parser.parse_args()
    
    # Auto-detect promised_lead_time if not provided
    promised_lead_time = args.promised_lead_time
    if promised_lead_time is None:
        detected = detect_promised_lead_time(args.directory)
        if detected is not None:
            promised_lead_time = detected
            print(f"Auto-detected promised_lead_time={promised_lead_time} from folder path")
        else:
            print("Error: --promised-lead-time not provided and could not auto-detect from path.")
            print("Please provide --promised-lead-time or use a folder with lead_time_0, lead_time_4, or lead_time_stochastic in path.")
            sys.exit(1)
    
    benchmark_all(
        promised_lead_time=promised_lead_time,
        instance_dir=args.directory,
        max_periods=args.max_periods,
        max_workers=args.max_workers,
        model_name=args.model
    )
