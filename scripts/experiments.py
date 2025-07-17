import subprocess
import time
import pandas as pd
import os
from datetime import datetime


TEST_CASES = [
    # Basic tests
    {
        "name": "input2D_100_100_100_4",
        "input_file": "../test_files/input2D.inp",
        "k": 100, 
        "max_iterations": 100,
        "change_threshold": 100,
        "move_threshold": 0.4,
        "output_file": "output2D_test.txt"
    },
    {
        "name": "input100D2_100_10_0_0005",
        "input_file": "../test_files/input100D2.inp",
        "k": 100,
        "max_iterations": 10, 
        "change_threshold": 0,
        "move_threshold": 0.005,
        "output_file": "output100D2_test.txt"
    },
    # K variation tests
    {
        "name": "input100D2_K10",
        "input_file": "../test_files/input100D2.inp",
        "k": 10,
        "max_iterations": 10, 
        "change_threshold": 0,
        "move_threshold": 0.005,
        "output_file": "output100D2_K10.txt"
    },
    {
        "name": "input100D2_K500",
        "input_file": "../test_files/input100D2.inp",
        "k": 500,
        "max_iterations": 10, 
        "change_threshold": 0,
        "move_threshold": 0.005,
        "output_file": "output100D2_K500.txt"
    },
    # Different dimensions test
    {
        "name": "input10D_100_10_0_0005",
        "input_file": "../test_files/input10D.inp",
        "k": 100,
        "max_iterations": 10, 
        "change_threshold": 0,
        "move_threshold": 0.005,
        "output_file": "output10D_test.txt"
    }
]

IMPLEMENTATIONS = [
    {"name": "Sequential", "executable": "../build/src/kmeans_seq", "args": []},
    {"name": "CUDA Centroids", "executable": "../build/src/kmeans_centroids", "args": []},
    {"name": "CUDA Distances", "executable": "../build/src/kmeans_distances", "args": []},
    {"name": "CUDA Full", "executable": "../build/src/kmeans_full", "args": []}
]


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_experiment(test_case, implementation):
    """Run a single experiment with the given test case and implementation"""
    cmd = [
        implementation["executable"],
        test_case["input_file"],
        str(test_case["k"]),
        str(test_case["max_iterations"]),
        str(test_case["change_threshold"]),
        str(test_case["move_threshold"]),
        test_case["output_file"]
    ]
    
    cmd.extend(implementation["args"])
    print(f"Running: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        runtime = time.time() - start_time
        
        if result.returncode != 0:
            print(f"Error running command: {' '.join(cmd)}")
            print(result.stderr)
            return {
                "test_case": test_case["name"],
                "implementation": implementation["name"],
                "runtime": -1,
                "error": result.stderr,
                "successful": False
            }
        
        return {
            "test_case": test_case["name"],
            "implementation": implementation["name"],
            "runtime": runtime,
            "successful": True
        }
    
    except Exception as e:
        print(f"Exception running command: {e}")
        return {
            "test_case": test_case["name"],
            "implementation": implementation["name"],
            "runtime": -1,
            "error": str(e),
            "successful": False
        }


def run_all_experiments():
    """Run all combinations of test cases and implementations"""
    all_results = []
    
    for test_case in TEST_CASES:
        print(f"\nRunning test case: {test_case['name']}")
        
        # Get baseline
        seq_impl = None
        for impl in IMPLEMENTATIONS:
            if impl["name"] == "Sequential":
                seq_impl = impl
                break

        seq_result = run_experiment(test_case, seq_impl)
        all_results.append(seq_result)
        
        if seq_result["successful"]:
            seq_time = seq_result["runtime"]
            
            # Other implementations
            for implementation in IMPLEMENTATIONS:
                if implementation["name"] == "Sequential":
                    continue
                
                result = run_experiment(test_case, implementation)
                
                # Calculate acceleration when successful
                if result["successful"]:
                    result["acceleration"] = seq_time / result["runtime"]
                else:
                    result["acceleration"] = 0
                
                all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"{RESULTS_DIR}/kmeans_results_{timestamp}.csv", index=False)
    
    return results_df


def print_table_results(results_df):
    """Print results in a simple table format in the terminal"""

    unique_test_cases = list(set(results_df['test_case']))
    
    for test_name in unique_test_cases:
        print("\n=== Results for " + test_name + " ===")
        test_results = results_df[results_df['test_case'] == test_name]

        print("Implementation\t\tRuntime(s)\t\tAcceleration")
        print("--------------------------------------------------")
        
        for index, row in test_results.iterrows():
            impl = row['implementation']
            
            if row['successful']:
                runtime = "{:.6f}".format(row['runtime'])
                if 'acceleration' in row:
                    accel = str(round(row['acceleration'], 2))
                else:
                    accel = "-"

            else:
                runtime = "ERROR"
                accel = "-"
                
            print(impl + "\t\t" + runtime + "\t\t" + accel)
        
        print("\n")


if __name__ == "__main__":
    results = run_all_experiments()
    # results = pd.read_csv(f"{RESULTS_DIR}/kmeans_results_20250507_093751.csv")
    print_table_results(results)
    print(f"\nResults saved to : {RESULTS_DIR}")