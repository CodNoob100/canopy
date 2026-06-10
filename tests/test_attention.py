import numpy as np
from canopy.nas.tropical_score import tropical_expressivity_score_darts, score_attention_regions
from canopy.nas.architecture import ArchitectureSpec, TransformerSearchSpace, TransformerSpec

def test_tighter_skip_connection_bound():
    """Verify that a skip connection increases the expressivity score."""
    # Create an architecture with standard convs
    arch_base = ArchitectureSpec(layer_widths=(), cell_ops=("sep_conv_3x3~0", "sep_conv_5x5~1"))
    score_base = tropical_expressivity_score_darts(arch_base, input_spatial_dim=32, base_channels=16)
    
    # Create the same architecture but with an added skip connection
    arch_skip = ArchitectureSpec(layer_widths=(), cell_ops=("sep_conv_3x3~0", "sep_conv_5x5~1", "skip_connect~2"))
    score_skip = tropical_expressivity_score_darts(arch_skip, input_spatial_dim=32, base_channels=16)
    
    assert score_skip > score_base, f"Expected skip connection to increase score. Base: {score_base}, Skip: {score_skip}"


def test_score_attention_regions_basic():
    """Test the score_attention_regions proxy with a small toy sequence."""
    N_seq = 4
    d_model = 16
    H_heads = 2
    
    # N=4 -> 4*3/2 = 6 hyperplanes per head
    # d_model = 16, H_heads = 2 -> d_k = 8
    # limit = min(8, 6) = 6
    # regions_per_query_per_head = sum(comb(6, i) for i in 0..6) = 2^6 = 64
    # log_regions_per_query_per_head = log10(64) ~ 1.806
    # total per query = 1.806 * 2 = 3.612
    # total for N_seq = 3.612 * 4 = 14.449
    
    total_log_regions = score_attention_regions(N_seq, d_model, H_heads)
    
    expected_log = np.log10(64) * H_heads * N_seq
    assert np.isclose(total_log_regions, expected_log, atol=1e-5)


def test_score_attention_regions_large():
    """Test the score_attention_regions proxy with large real-world sequence to ensure no overflow."""
    N_seq = 1024
    d_model = 768
    H_heads = 12
    
    total_log_regions = score_attention_regions(N_seq, d_model, H_heads)
    
    # Should be a very large positive number (log10 of regions), but not infinite
    assert total_log_regions > 1000
    assert not np.isinf(total_log_regions)
    assert not np.isnan(total_log_regions)


def test_transformer_search_space():
    """Test the new TransformerSearchSpace architecture generation and mutations."""
    rng = np.random.default_rng(42)
    search_space = TransformerSearchSpace(min_depth=2, max_depth=4, min_d_model=128, max_d_model=256, allowed_heads=(4, 8), N_seq=128)
    
    # Test random generation
    arch1 = search_space.random(rng)
    assert isinstance(arch1, TransformerSpec)
    assert 2 <= len(arch1.cell_ops) <= 4
    assert 128 <= arch1.layer_widths[0] <= 256
    
    # Test mutation
    arch2 = search_space.mutate(arch1, rng)
    assert isinstance(arch2, TransformerSpec)
    assert 2 <= len(arch2.cell_ops) <= 4
    assert 128 <= arch2.layer_widths[0] <= 256
    
    # Test crossover
    arch3 = search_space.crossover(arch1, arch2, rng)
    assert isinstance(arch3, TransformerSpec)
    assert arch3.layer_widths[0] in [arch1.layer_widths[0], arch2.layer_widths[0]]
