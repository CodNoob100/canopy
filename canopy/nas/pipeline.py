import logging
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ArchitectureCandidate:
    arch_id: str
    arch_str: str
    tropical_score: float = 0.0
    proxy_accuracy: float = 0.0
    final_accuracy: float = 0.0


class MultiStagePipeline:
    """
    A multi-stage NAS pipeline designed for massive search spaces (e.g., DARTS / NAS-Bench-301).
    
    Stage 1: Tropical Pre-filter (Training-Free, Score entire pool, keep Top-K)
    Stage 2: Proxy Training (Lightweight training on survivors)
    Stage 3: Full Training (End-to-end evaluation on finalists)
    """
    def __init__(self, 
                 score_fn: Callable[[str], float],
                 proxy_train_fn: Callable[[str], float],
                 full_train_fn: Callable[[str], float],
                 filter_keep_ratio: float = 0.1,
                 proxy_keep_top_k: int = 5):
        """
        Args:
            score_fn: Function mapping arch_str to Tropical Score (Generic/Tight).
            proxy_train_fn: Function mapping arch_str to Proxy validation accuracy (e.g., 5 epochs).
            full_train_fn: Function mapping arch_str to Final test accuracy (e.g., 100 epochs).
            filter_keep_ratio: The fraction of architectures to keep after Stage 1.
            proxy_keep_top_k: The absolute number of architectures to fully train after Stage 2.
        """
        self.score_fn = score_fn
        self.proxy_train_fn = proxy_train_fn
        self.full_train_fn = full_train_fn
        self.filter_keep_ratio = filter_keep_ratio
        self.proxy_keep_top_k = proxy_keep_top_k

    def run(self, search_space_candidates: List[str]) -> List[ArchitectureCandidate]:
        """Runs the 3-stage NAS pipeline."""
        candidates = [ArchitectureCandidate(arch_id=str(i), arch_str=arch) for i, arch in enumerate(search_space_candidates)]
        total = len(candidates)
        
        # ==========================================
        # STAGE 1: TROPICAL PRE-FILTER
        # ==========================================
        logger.info(f"--- Stage 1: Tropical Pre-Filter ({total} candidates) ---")
        for c in candidates:
            c.tropical_score = self.score_fn(c.arch_str)
            
        # Sort descending by expressivity
        candidates.sort(key=lambda x: x.tropical_score, reverse=True)
        keep_count = max(1, int(total * self.filter_keep_ratio))
        survivors_s1 = candidates[:keep_count]
        logger.info(f"Stage 1 complete. Filtered {total} -> {keep_count} candidates.")

        # ==========================================
        # STAGE 2: PROXY TRAINING
        # ==========================================
        logger.info(f"--- Stage 2: Proxy Training ({keep_count} candidates) ---")
        for c in survivors_s1:
            c.proxy_accuracy = self.proxy_train_fn(c.arch_str)
            
        # Sort descending by proxy accuracy
        survivors_s1.sort(key=lambda x: x.proxy_accuracy, reverse=True)
        finalists = survivors_s1[:self.proxy_keep_top_k]
        logger.info(f"Stage 2 complete. Filtered {keep_count} -> {len(finalists)} finalists.")

        # ==========================================
        # STAGE 3: FULL TRAINING
        # ==========================================
        logger.info(f"--- Stage 3: Full Training ({len(finalists)} finalists) ---")
        for c in finalists:
            c.final_accuracy = self.full_train_fn(c.arch_str)
            logger.info(f"Finalist {c.arch_id} scored: {c.final_accuracy:.2f}%")

        # Sort by final accuracy
        finalists.sort(key=lambda x: x.final_accuracy, reverse=True)
        logger.info(f"Pipeline complete. Best architecture: {finalists[0].arch_id} with {finalists[0].final_accuracy:.2f}%")
        
        return finalists
