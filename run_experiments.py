import os
import re
import json
from scen1_negotiation import _build_model, run_negotiation

LOG_DIR = "conversation_logs"

def parse_price(price_str: str) -> float | None:
    """Convert price string like '$11,500' to float 11500.0"""
    if not price_str:
        return None
    cleaned = re.sub(r'[^\d.]', '', price_str)
    try:
        return float(cleaned)
    except ValueError:
        return None

def run_experiment(name_prefix: str, buyer_name: str, seller_name: str, monitor_name: str, iterations: int = 5):
    """Run an experiment multiple times and return the stats."""
    print(f"\n=======================================================")
    print(f"Starting Experiment: {name_prefix} (Buyer: {buyer_name}, Seller: {seller_name}, Monitor: {monitor_name})")
    print(f"=======================================================\n")
    
    buyer_model = _build_model(buyer_name)
    seller_model = _build_model(seller_name)
    monitor_model = _build_model(monitor_name)
    
    deals_reached = 0
    walk_aways = 0
    max_turns = 0
    total_deceptions = 0
    total_price = 0.0

    for i in range(iterations):
        print(f"--- Running {name_prefix}_{i} ---")
        outcome, price_str, transcript, deceptions = run_negotiation(buyer_model, seller_model, monitor_model)
        
        # Save transcript
        log_file = os.path.join(LOG_DIR, f"{name_prefix}_{i}.txt")
        with open(log_file, "w") as f:
            f.write(transcript)
            
        print(f"Completed {name_prefix}_{i}: Outcome={outcome}, Price={price_str}")
        
        if outcome == "deal_reached":
            deals_reached += 1
            price_val = parse_price(price_str)
            if price_val is not None:
                total_price += price_val
        elif outcome == "walk_away":
            walk_aways += 1
        elif outcome == "max_turns":
            max_turns += 1

        total_deceptions += deceptions

    avg_price = (total_price / deals_reached) if deals_reached > 0 else 0.0
    
    return {
        "deals_reached": deals_reached,
        "walk_aways": walk_aways,
        "max_turns": max_turns,
        "total_deceptions": total_deceptions,
        "avg_price": avg_price
    }

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. 5 runs of Kimi measuring Kimi (K2K2)
    k2k2_stats = run_experiment(
        name_prefix="k2k2",
        buyer_name="kimi",
        seller_name="kimi",
        monitor_name="llama8b",
        iterations=3
    )
    
    # 2. 5 runs of GPT-OSS measuring Kimi (K2GPT)
    gptk2_stats = run_experiment(
        name_prefix="gptk2",
        buyer_name="gpt20b",
        seller_name="kimi",
        monitor_name="llama8b",
        iterations=3
    )
    
    # Generate overall stats summary
    summary_text = (
        "====================================\n"
        "      EXPERIMENT RESULTS SUMMARY      \n"
        "====================================\n\n"
        "K2K2 (Buyer: Kimi, Seller: Kimi)\n"
        "------------------------------------\n"
        f"Deals Reached: {k2k2_stats['deals_reached']}\n"
        f"Walk Aways:    {k2k2_stats['walk_aways']}\n"
        f"Max Turns:     {k2k2_stats['max_turns']}\n"
        f"Deceptions:    {k2k2_stats['total_deceptions']}\n"
        f"Average Price: ${k2k2_stats['avg_price']:.2f}\n\n"
        
        "K2GPT (Buyer: GPT-20B, Seller: Kimi)\n"
        "------------------------------------\n"
        f"Deals Reached: {k2gpt_stats['deals_reached']}\n"
        f"Walk Aways:    {k2gpt_stats['walk_aways']}\n"
        f"Max Turns:     {k2gpt_stats['max_turns']}\n"
        f"Deceptions:    {k2gpt_stats['total_deceptions']}\n"
        f"Average Price: ${k2gpt_stats['avg_price']:.2f}\n"
    )
    
    summary_file = os.path.join(LOG_DIR, "results.txt")
    with open(summary_file, "w") as f:
        f.write(summary_text)
        
    print("\n" + summary_text)
    print(f"Done. Summaries and logs saved to ./{LOG_DIR}/")

if __name__ == "__main__":
    main()
