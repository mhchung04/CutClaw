import os
import json
import time
import threading
from datetime import datetime
import litellm

# Lock for thread safety
_log_lock = threading.Lock()

COST_LOG_FILE = "litellm_costs.log"
SUMMARY_JSON_FILE = "litellm_cost_summary.json"

def get_output_paths():
    # Attempt to read from config.VIDEO_DATABASE_FOLDER
    try:
        from src import config
        base_dir = getattr(config, "VIDEO_DATABASE_FOLDER", "./Output")
    except Exception:
        base_dir = "./Output"
    
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, COST_LOG_FILE), os.path.join(base_dir, SUMMARY_JSON_FILE)

def track_cost_callback(kwargs, completion_response, start_time, end_time):
    try:
        model = kwargs.get("model", "unknown")
        
        # Calculate cost
        cost = 0.0
        if "response_cost" in kwargs and kwargs["response_cost"] is not None:
            cost = float(kwargs["response_cost"])
        else:
            try:
                cost = litellm.completion_cost(completion_response=completion_response)
            except Exception:
                cost = 0.0
        
        # Handle usage details
        usage = getattr(completion_response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0
        else:
            # Check if it's a dict
            if isinstance(completion_response, dict) and "usage" in completion_response:
                u = completion_response["usage"]
                prompt_tokens = u.get("prompt_tokens", 0)
                completion_tokens = u.get("completion_tokens", 0)
                total_tokens = u.get("total_tokens", 0)
            else:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                
        duration = 0.0
        if start_time and end_time:
            if hasattr(start_time, "timestamp") and hasattr(end_time, "timestamp"):
                duration = (end_time - start_time).total_seconds()
            else:
                try:
                    duration = float(end_time - start_time)
                except Exception:
                    pass

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        log_path, summary_path = get_output_paths()
        
        with _log_lock:
            # 1. Append to the human-readable log file
            log_line = (
                f"[{timestamp_str}] Model: {model} | Cost: ${cost:.6f} | "
                f"Tokens: In={prompt_tokens}, Out={completion_tokens}, Total={total_tokens} | "
                f"Duration: {duration:.2f}s\n"
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_line)
            
            # Print beautiful cost info to standard output/console
            print(
                f"\n💰 [LiteLLM Call Cost] Model: {model} | Cost: ${cost:.6f} | "
                f"Tokens: {prompt_tokens} (In) / {completion_tokens} (Out) / {total_tokens} (Total) | "
                f"Duration: {duration:.2f}s"
            )
            
            # 2. Update JSON summary file (per-model statistics)
            summary_data = {"total_cost_usd": 0.0, "models": {}}
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary_data = json.load(f)
                except Exception:
                    pass
            
            # Ensure proper keys exist
            if "total_cost_usd" not in summary_data:
                summary_data["total_cost_usd"] = 0.0
            if "models" not in summary_data:
                summary_data["models"] = {}
                
            # Update cumulative total cost
            summary_data["total_cost_usd"] += cost
            
            # Update per-model stats
            models_dict = summary_data["models"]
            if model not in models_dict:
                models_dict[model] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0
                }
            
            m_stats = models_dict[model]
            m_stats["calls"] = m_stats.get("calls", 0) + 1
            m_stats["prompt_tokens"] = m_stats.get("prompt_tokens", 0) + prompt_tokens
            m_stats["completion_tokens"] = m_stats.get("completion_tokens", 0) + completion_tokens
            m_stats["total_tokens"] = m_stats.get("total_tokens", 0) + total_tokens
            m_stats["cost_usd"] = m_stats.get("cost_usd", 0.0) + cost
            
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
                
            print(
                f"📈 [LiteLLM Cumulative Cost] Overall Total: ${summary_data['total_cost_usd']:.6f} | "
                f"'{model}' Total: ${m_stats['cost_usd']:.6f} ({m_stats['calls']} calls)\n"
            )
                  
    except Exception as e:
        print(f"⚠️ [LiteLLM Cost Tracker] Error logging cost: {e}")

# Register custom callback globally on litellm
if not hasattr(litellm, "success_callback") or litellm.success_callback is None:
    litellm.success_callback = []
    
if isinstance(litellm.success_callback, list):
    if track_cost_callback not in litellm.success_callback:
        litellm.success_callback.append(track_cost_callback)
else:
    # If success_callback is some other type (e.g. string or single function, though normally a list)
    try:
        litellm.success_callback = [track_cost_callback]
    except Exception:
        pass
