from src.env.environment import MECEnvironment
from src.utils.metrics import compute_average_latency


def main() -> None:
    env = MECEnvironment(
        num_users=5,
        num_servers=3,
        area_size=100,
        user_speed=3.0,
        server_capacity=5,
        seed=42,
    )

    state = env.reset()
    print("=== Initial State ===")
    print(state)
    print("Initial avg latency:", round(compute_average_latency(state), 4))

    for step in range(5):
        state = env.step()
        avg_latency = compute_average_latency(state)
        print(f"\n=== Step {step + 1} ===")
        print("Avg latency:", round(avg_latency, 4))
        for user in state["users"]:
            print(
                f"User {user['user_id']} -> "
                f"pos={user['position']} "
                f"server={user['connected_server_id']}"
            )


if __name__ == "__main__":
    main()