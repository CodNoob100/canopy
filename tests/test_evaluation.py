import os
import pandas as pd
from canopy.nas.architecture import NASBench201SearchSpace
from canopy.nas.benchmark import SyntheticBenchmark
from canopy.evaluation.runner import ExperimentRunner

def test_experiment_runner(tmp_path):
    space = NASBench201SearchSpace()
    bench = SyntheticBenchmark(n_architectures=100)
    
    # Use tmp_path for output
    runner = ExperimentRunner(
        search_space=space,
        benchmark=bench,
        strategies=["tropical_generic", "random"],
        population_size=10,
        n_generations=2,
        max_evaluations=20,
        output_dir=str(tmp_path)
    )
    
    results = runner.run_all(n_seeds=1)
    
    assert len(results) == 1
    res = results[0]["results"]
    assert "tropical_generic" in res
    assert "random" in res
    
    assert res["tropical_generic"]["total_evaluations"] == 20
    assert len(res["tropical_generic"]["trajectory_accuracy"]) > 0
    
    # Check if files were created
    csv_files = list(tmp_path.glob("*.csv"))
    json_files = list(tmp_path.glob("*.json"))
    assert len(csv_files) == 1
    assert len(json_files) == 2
    manifest_file = next(tmp_path.glob("experiment_manifest_*.json"))
    with open(manifest_file, encoding="utf-8") as f:
        import json
        manifest = json.load(f)
    assert manifest["benchmark"]["kind"] == "synthetic"
    assert manifest["benchmark"]["accuracy_source"] == "synthetic_simulation"
    assert manifest["configuration"]["seeds"] == [0]
    assert manifest["configuration"]["fitness"]["tropical_generic"]["units"] == "count_bound"
    assert (tmp_path / manifest["files"]["results_json"]).exists()
    assert (tmp_path / manifest["files"]["trajectory_csv"]).exists()
    
    # Verify CSV format
    df = pd.read_csv(csv_files[0])
    assert "seed" in df.columns
    assert "strategy" in df.columns
    assert "evaluations" in df.columns
    assert "best_test_accuracy" in df.columns
    assert len(df) > 0
