"""csat-forge: verified, parametric Korean CSAT (수능) math problem generator."""
from csat_forge.problem_engine import GENERATORS, KILLER_TOPICS, generate, generate_batch, verify
from csat_forge.meta_killer_engine import generate_meta, generate_meta_batch

__all__ = ["GENERATORS", "KILLER_TOPICS", "generate", "generate_batch", "verify",
           "generate_meta", "generate_meta_batch"]
__version__ = "0.2.0"
