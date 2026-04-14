import numpy as np
import pybullet as p
from factory_gym_env import FactoryGymEnv

def run_tests():
    results = {}
    print("Starting Environment Sanity Check...")

    # Test 1 - Creation
    try:
        env = FactoryGymEnv(render=False)
        results["Test 1 Environment Creation"] = "PASSED"
    except Exception as e:
        print(f"Test 1 FAILED: {str(e)}")
        results["Test 1 Environment Creation"] = "FAILED"
        return # Cannot continue if creation fails

    # Test 2 - Reset
    try:
        obs, _ = env.reset()
        if not isinstance(obs, np.ndarray):
            print("Test 2 FAILED: Reset did not return a numpy array")
            results["Test 2 Reset"] = "FAILED"
        elif obs.shape != (13,):
            print(f"Test 2 FAILED: Obs shape is {obs.shape}, expected (13,)")
            results["Test 2 Reset"] = "FAILED"
        elif np.all(obs == 0):
            print("Test 2 FAILED: Observation contains all zeros")
            results["Test 2 Reset"] = "FAILED"
        else:
            results["Test 2 Reset"] = "PASSED"
    except Exception as e:
        print(f"Test 2 FAILED with exception: {str(e)}")
        results["Test 2 Reset"] = "FAILED"

    # Test 3 & 4 - Step Logic and Info Dict
    rewards = []
    observations = []
    info_keys_passed = True
    missing_key = ""
    done_at_step = -1
    
    required_info_keys = {"cycle_time", "distance", "throughput", "idle_time", "task", "reward"}

    try:
        print("\nRunning transition steps...")
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            rewards.append(reward)
            observations.append(obs.copy())

            # Test 4 - Info dict check
            if info_keys_passed:
                for key in required_info_keys:
                    if key not in info:
                        info_keys_passed = False
                        missing_key = key
            
            # Test 5 - Done flag check
            if (terminated or truncated) and done_at_step == -1:
                done_at_step = i

            # Progress print every 100 steps
            if (i + 1) % 100 == 0:
                print(f"Step {i+1}: MinObs={np.min(obs):.2f}, MaxObs={np.max(obs):.2f}, MeanObs={np.mean(obs):.2f}, Reward={reward:.2f}")

        # Evaluate Test 3
        if len(set(rewards)) <= 1:
            print(f"Test 3 FAILED: Reward is constant ({rewards[0]})")
            results["Test 3 Step random actions"] = "FAILED"
        # Check if observations are changing (look for non-zero variance)
        elif np.all(np.std(observations, axis=0) < 1e-6):
            print("Test 3 FAILED: Observations are stuck/static")
            results["Test 3 Step random actions"] = "FAILED"
        else:
            results["Test 3 Step random actions"] = "PASSED"

        # Evaluate Test 4
        if info_keys_passed:
            results["Test 4 Info dict keys"] = "PASSED"
        else:
            print(f"Test 4 FAILED: Missing key '{missing_key}'")
            results["Test 4 Info dict keys"] = "FAILED"

        # Evaluate Test 5
        if 0 < done_at_step < 500:
            results["Test 5 Done flag"] = "PASSED"
        else:
            results["Test 5 Done flag"] = "WARNING"
            print(f"Test 5 WARNING: Episode never ended within 500 steps (Done observed at step: {done_at_step})")

        # Test 6 - Reset after done
        try:
            obs, _ = env.reset()
            if isinstance(obs, np.ndarray) and not np.all(obs == 0):
                results["Test 6 Reset after done"] = "PASSED"
            else:
                print("Test 6 FAILED: Invalid observation after post-done reset")
                results["Test 6 Reset after done"] = "FAILED"
        except Exception as e:
            print(f"Test 6 FAILED: {str(e)}")
            results["Test 6 Reset after done"] = "FAILED"

    except Exception as e:
        print(f"Step loop FAILED: {str(e)}")
        results["Test 3 Step random actions"] = "FAILED"

    # Final Summary
    print("\n=================== SANITY CHECK RESULTS ===================")
    overall_passed = True
    # Correct key ordering for display matching user request
    test_keys = [
        "Test 1 Environment Creation",
        "Test 2 Reset",
        "Test 3 Step random actions",
        "Test 4 Info dict keys",
        "Test 5 Done flag",
        "Test 6 Reset after done"
    ]
    
    for test in test_keys:
        res = results.get(test, "FAILED")
        print(f"{test:<30}: {res}")
        if res == "FAILED": overall_passed = False
    
    if overall_passed:
        print("Overall                      : ALL TESTS PASSED — ready for PPO training")
    else:
        print("Overall                      : NOT READY FOR TRAINING — fix the issues above")
    print("============================================================\n")

if __name__ == "__main__":
    run_tests()
