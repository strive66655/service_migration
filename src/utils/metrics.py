from typing import Dict

def compute_average_latency(state: Dict) -> float:
    total_latency = 0.0
    count = 0

    for user in state["users"]:
        server_id = user["connected_server_id"]
        if server_id is not None:
            total_latency += user["latency_to_servers"][server_id]
            count += 1
        
    return total_latency / count if count > 0 else 0.0