import argparse
import os
import re
import json

import fcntl

from scen1_negotiation import _build_model, _build_monitor_model, run_negotiation

# Old code is below

LOG_DIR = "experiments/conversation_logs.json"
LOCK_FILE = f"{LOG_DIR}.lock"
MAX_FAILED_ATTEMPTS_PER_ITERATION = int(os.getenv("MAX_FAILED_ATTEMPTS_PER_ITERATION", "5"))


EXPERIMENT_MATRIX = [

    # {
    #     "name": "qwen27b-qwen235",
    #     "buyer": "qwen27b",
    #     "seller": "qwen235b",
    #     "monitor": "qwen2b",
    #     "target_iterations": 100
    # },

    {
        "name": "kimi26-kimi26",
        "buyer": "kimi",
        "seller": "kimi",
        "monitor": "qwen2b",
        "seller_monitoring_notice": False,
        "target_iterations": 100,
    }


    # {
    #     "name": "aware_qwen27b-qwen235",
    #     "aliases": ["qaware_qwen27b-qwen235"],
    #     "buyer": "qwen27b",
    #     "seller": "qwen235b",
    #     "monitor": "qwen2b",
    #     "seller_monitoring_notice": True,
    #     "target_iterations": 100
    # },


    # {
    #     "name": "None_qwen27b-qwen235",
    #     "buyer": "qwen27b",
    #     "seller": "qwen235b",
    #     "monitor": "none",
    #     "target_iterations": 100
    # },
    

    # {
    #     "name": "None_qwen2b-qwen2b", 
    #     "buyer": "qwen2b", 
    #     "seller": "qwen2b", 
    #     "monitor": "none", 
    #     "target_iterations": 100
    # },

    # {
    #     "name": "qwen2b-qwen2b", 
    #     "buyer": "qwen2b", 
    #     "seller": "qwen2b", 
    #     "monitor": "qwen2b", 
    #     "target_iterations": 100
    # },
    # {
    #     "name": "qwen2b-qwen235",
    #     "buyer": "qwen2b",
    #     "seller": "qwen235b",
    #     "monitor": "qwen2b",
    #     "target_iterations": 100
    # },

   
    # {
    #     "name": "None_qwen2b-qwen235",
    #     "buyer": "qwen2b",
    #     "seller": "qwen235b",
    #     "monitor": "none",
    #     "target_iterations": 100
    # },

    # {
    #     "name": "qwen235-qwen235",
    #     "buyer": "qwen235b",
    #     "seller": "qwen235b",
    #     "monitor": "qwen2b",
    #     "target_iterations": 100
    # },


    # {
    #     "name": "K2GPT", 
    #     "buyer": "gpt20b", 
    #     "seller": "kimi", 
    #     "monitor": "qwen2b", 
    #     "target_iterations": 50
    # }
]

def parse_price(price_str: str) -> float | None:
    """Convert price string like '$11,500' to float 11500.0"""
    if not price_str:
        return None
    cleaned = re.sub(r'[^\d.]', '', price_str)
    try:
        return float(cleaned)
    except ValueError:
        return None

def load_state() -> dict:
    """Loads the state file so we can resume where we left off."""
    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)
    with open(LOCK_FILE, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        try:
            return _read_state_unlocked()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read_state_unlocked() -> dict:
    if os.path.exists(LOG_DIR):
        with open(LOG_DIR, "r") as f:
            return json.load(f)
    return {}


def init_experiment_data(config: dict) -> dict:
    return {
        "buyer_model": config["buyer"],
        "seller_model": config["seller"],
        "monitor_model": config["monitor"],
        "seller_monitoring_notice": config.get("seller_monitoring_notice", False),
        "completed_iterations": 0,
        "deals_reached": 0,
        "walk_aways": 0,
        "max_turns": 0,
        "total_deceptions": 0,
        "total_price_sum": 0.0,
        "avg_price": 0.0,
    }


def legacy_state_key(config: dict) -> str:
    awareness_prefix = "aware-" if config.get("seller_monitoring_notice", False) else ""
    log_folder = os.path.join("experiments", f"{config['buyer']}_{config['seller']}")
    return os.path.join(log_folder, f"{awareness_prefix}{config['monitor']}.json")


def state_entry_matches_config(entry: dict, config: dict) -> bool:
    return (
        entry.get("buyer_model") == config["buyer"]
        and entry.get("seller_model") == config["seller"]
        and entry.get("monitor_model") == config["monitor"]
        and bool(entry.get("seller_monitoring_notice", False))
        == bool(config.get("seller_monitoring_notice", False))
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run negotiation experiment batches.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Only run these experiment names. Use names from EXPERIMENT_MATRIX.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Restart selected experiments from iteration 0.",
    )
    parser.add_argument(
        "--target-iterations",
        type=int,
        help="Override target_iterations for selected experiments.",
    )
    return parser.parse_args(argv)


def selected_experiments(args: argparse.Namespace) -> list[dict]:
    if not args.experiments:
        configs = [dict(config) for config in EXPERIMENT_MATRIX]
    else:
        requested = set(args.experiments)
        configs = [
            dict(config)
            for config in EXPERIMENT_MATRIX
            if config["name"] in requested or requested.intersection(config.get("aliases", []))
        ]
        found = {
            requested_name
            for config in configs
            for requested_name in (config["name"], *config.get("aliases", []))
            if requested_name in requested
        }
        missing = sorted(requested - found)
        if missing:
            raise SystemExit(f"Unknown experiment name(s): {', '.join(missing)}")

    if args.target_iterations is not None:
        for config in configs:
            config["target_iterations"] = args.target_iterations

    return configs


def save_experiment_state(exp_name: str, exp_state: dict, config: dict | None = None):
    """Save only one experiment section while preserving parallel writers' data."""
    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)
    with open(LOCK_FILE, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            latest_state = _read_state_unlocked()
            if config is not None:
                for key, value in list(latest_state.items()):
                    if key != exp_name and isinstance(value, dict) and state_entry_matches_config(value, config):
                        latest_state.pop(key)
            latest_state[exp_name] = exp_state
            temp_path = f"{LOG_DIR}.tmp.{os.getpid()}"
            with open(temp_path, "w") as f:
                json.dump(latest_state, f, indent=4)
            os.replace(temp_path, LOG_DIR)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

def main(argv: list[str] | None = None):
    args = parse_args(argv)
    print("=======================================================")
    print("  STARTING LARGE-SCALE NEGOTIATION EXPERIMENT PIPELINE ")
    print("=======================================================\n")

    # 1. Load existing progress
    state = load_state()
    configs = selected_experiments(args)

    # 2. Iterate through the experiment matrix
    for config in configs:
        exp_name = config["name"]
        target = config["target_iterations"]

        if args.reset:
            exp_data = init_experiment_data(config)
            print(f"Resetting experiment: {exp_name}")
        elif exp_name in state:
            exp_data = state[exp_name]
            print(f"Resuming {exp_name} from {LOG_DIR}...")
        elif legacy_state_key(config) in state:
            exp_data = state[legacy_state_key(config)]
            print(f"Migrating legacy state for {exp_name} from {legacy_state_key(config)}...")
        else:
            exp_data = init_experiment_data(config)
            print(f"Starting new experiment: {exp_name}")

        completed = exp_data["completed_iterations"]

        if completed >= target:
            print(f"Skipping {exp_name}: Already completed {completed}/{target} runs.\n")
            continue

        # Load models only when we are sure we need to run this experiment
        buyer_model = _build_model(config["buyer"])
        seller_model = _build_model(config["seller"])
        monitor_model = _build_monitor_model(config["monitor"])

        while exp_data["completed_iterations"] < target:
            i = exp_data["completed_iterations"]
            failures = 0

            while True:
                run_id = str(i) if failures == 0 else f"{i}_retry{failures}"
                print(f"\n--- {exp_name}: Running Iteration {i} (attempt {failures + 1}) ---")

                try:
                    outcome, price_str, transcript, deceptions = run_negotiation(
                        buyer_model,
                        seller_model,
                        monitor_model,
                        run_id=run_id,
                        exit_on_error=False,
                        seller_monitoring_notice=config.get("seller_monitoring_notice", False),
                        artifact_namespace=exp_name,
                    )
                    break
                except Exception as e:
                    failures += 1
                    print(f" Iteration {i} failed: {e}")
                    if failures >= MAX_FAILED_ATTEMPTS_PER_ITERATION:
                        raise RuntimeError(f"{exp_name} iteration {i} failed {MAX_FAILED_ATTEMPTS_PER_ITERATION} times; aborting.")
                    print(f" Retrying {exp_name} iteration {i}...")
            
            # Update stats
            exp_data["completed_iterations"] += 1
            exp_data["total_deceptions"] += deceptions
            
            if outcome == "deal_reached":
                exp_data["deals_reached"] += 1
                price_val = parse_price(price_str)
                if price_val is not None:
                    exp_data["total_price_sum"] += price_val
            elif outcome == "walk_away":
                exp_data["walk_aways"] += 1
            elif outcome == "max_turns":
                exp_data["max_turns"] += 1

            if exp_data["deals_reached"] > 0:
                exp_data["avg_price"] = exp_data["total_price_sum"] / exp_data["deals_reached"]

            save_experiment_state(exp_name, exp_data, config)
            print(f" Saved progress for {exp_name}_{i} to {LOG_DIR}")


if __name__ == "__main__":
    main()
















# LOG_DIR = "conversation_logs"

# def run_experiment(name_prefix: str, buyer_name: str, seller_name: str, monitor_name: str, iterations: int = 5):
#     """Run an experiment multiple times and return the stats."""
#     print(f"\n=======================================================")
#     print(f"Starting Experiment: {name_prefix} (Buyer: {buyer_name}, Seller: {seller_name}, Monitor: {monitor_name})")
#     print(f"=======================================================\n")
    
#     buyer_model = _build_model(buyer_name)
#     seller_model = _build_model(seller_name)
#     monitor_model = _build_model(monitor_name)
    
#     deals_reached = 0
#     walk_aways = 0
#     max_turns = 0
#     total_deceptions = 0
#     total_price = 0.0

#     for i in range(iterations):
#         print(f"--- Running {name_prefix}_{i} ---")
#         outcome, price_str, transcript, deceptions = run_negotiation(buyer_model, seller_model, monitor_model)
        
#         # Save transcript
#         log_file = os.path.join(LOG_DIR, f"{name_prefix}_{i}.txt")
#         with open(log_file, "w") as f:
#             f.write(transcript)
            
#         print(f"Completed {name_prefix}_{i}: Outcome={outcome}, Price={price_str}")
        
#         if outcome == "deal_reached":
#             deals_reached += 1
#             price_val = parse_price(price_str)
#             if price_val is not None:
#                 total_price += price_val
#         elif outcome == "walk_away":
#             walk_aways += 1
#         elif outcome == "max_turns":
#             max_turns += 1

#         total_deceptions += deceptions

#     avg_price = (total_price / deals_reached) if deals_reached > 0 else 0.0
    
#     return {
#         "deals_reached": deals_reached,
#         "walk_aways": walk_aways,
#         "max_turns": max_turns,
#         "total_deceptions": total_deceptions,
#         "avg_price": avg_price
#     }

# def main():
#     os.makedirs(LOG_DIR, exist_ok=True)
    
#     # 1. 5 runs of Kimi measuring Kimi (K2K2)
#     k2k2_stats = run_experiment(
#         name_prefix="k2k2",
#         buyer_name="kimi",
#         seller_name="kimi",
#         monitor_name="llama8b",
#         iterations=3
#     )
    
#     # 2. 5 runs of GPT-OSS measuring Kimi (K2GPT)
#     gptk2_stats = run_experiment(
#         name_prefix="gptk2",
#         buyer_name="gpt20b",
#         seller_name="kimi",
#         monitor_name="llama8b",
#         iterations=3
#     )
    
#     # Generate overall stats summary
#     summary_text = (
#         "====================================\n"
#         "      EXPERIMENT RESULTS SUMMARY      \n"
#         "====================================\n\n"
#         "K2K2 (Buyer: Kimi, Seller: Kimi)\n"
#         "------------------------------------\n"
#         f"Deals Reached: {k2k2_stats['deals_reached']}\n"
#         f"Walk Aways:    {k2k2_stats['walk_aways']}\n"
#         f"Max Turns:     {k2k2_stats['max_turns']}\n"
#         f"Deceptions:    {k2k2_stats['total_deceptions']}\n"
#         f"Average Price: ${k2k2_stats['avg_price']:.2f}\n\n"
        
#         "K2GPT (Buyer: GPT-20B, Seller: Kimi)\n"
#         "------------------------------------\n"
#         f"Deals Reached: {k2gpt_stats['deals_reached']}\n"
#         f"Walk Aways:    {k2gpt_stats['walk_aways']}\n"
#         f"Max Turns:     {k2gpt_stats['max_turns']}\n"
#         f"Deceptions:    {k2gpt_stats['total_deceptions']}\n"
#         f"Average Price: ${k2gpt_stats['avg_price']:.2f}\n"
#     )
    
#     summary_file = os.path.join(LOG_DIR, "results.txt")
#     with open(summary_file, "w") as f:
#         f.write(summary_text)
        
#     print("\n" + summary_text)
#     print(f"Done. Summaries and logs saved to ./{LOG_DIR}/")

# if __name__ == "__main__":
#     main()
