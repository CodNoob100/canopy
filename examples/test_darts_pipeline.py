import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from canopy.nas.pipeline import MultiStagePipeline
from canopy.nas.tropical_score import tropical_expressivity_score_darts
from canopy.nas.architecture import ArchitectureSpec

logging.basicConfig(level=logging.INFO)

# Dummy architectures simulating NAS-Bench-301 DARTS space
darts_candidates = [
    ("sep_conv_3x3", "skip_connect", "max_pool_3x3", "none", "sep_conv_5x5", "skip_connect", "none", "sep_conv_3x3", "dil_conv_3x3", "skip_connect", "max_pool_3x3", "sep_conv_5x5", "none", "none"),
    ("skip_connect", "none", "none", "sep_conv_3x3", "skip_connect", "max_pool_3x3", "sep_conv_3x3", "none", "sep_conv_5x5", "skip_connect", "none", "sep_conv_3x3", "dil_conv_3x3", "none"),
    ("sep_conv_5x5", "sep_conv_5x5", "sep_conv_3x3", "skip_connect", "max_pool_3x3", "skip_connect", "dil_conv_5x5", "none", "none", "skip_connect", "max_pool_3x3", "none", "none", "skip_connect")
]

# Create ArchitectureSpecs
arch_specs = {str(i): ArchitectureSpec(layer_widths=(), cell_ops=ops) for i, ops in enumerate(darts_candidates)}

def dummy_score_fn(arch_str: str) -> float:
    # Use our new DARTS tropical scoring proxy
    arch = arch_specs[arch_str]
    return tropical_expressivity_score_darts(arch)

def dummy_proxy_train(arch_str: str) -> float:
    # Simulate proxy training accuracy based on tropical score + noise
    score = dummy_score_fn(arch_str)
    return score * 0.5 + 40.0  # Just a dummy mapping

def dummy_full_train(arch_str: str) -> float:
    score = dummy_score_fn(arch_str)
    return score * 0.7 + 60.0

if __name__ == "__main__":
    print("Validating Multi-Stage DARTS Pipeline...")
    pipeline = MultiStagePipeline(
        score_fn=dummy_score_fn,
        proxy_train_fn=dummy_proxy_train,
        full_train_fn=dummy_full_train,
        filter_keep_ratio=0.5,
        proxy_keep_top_k=1
    )
    
    # Pass arch IDs as strings
    candidate_ids = [str(i) for i in range(len(darts_candidates))]
    finalists = pipeline.run(candidate_ids)
    
    print("\n[VALIDATION SUCCESS] Pipeline successfully ran and selected:", finalists[0].arch_id)
