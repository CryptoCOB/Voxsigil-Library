#!/usr/bin/env python
import logging
import time
import hashlib
import datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

# Configure logger
logger = logging.getLogger(__name__)

# Import semantic cache
from VoxSigilRag.voxsigil_semantic_cache import SemanticCacheManager

# Import hybrid middleware for improved version
try:
    from VoxSigilRag.hybrid_blt import HybridMiddleware
    HAS_HYBRID_MIDDLEWARE = True
except ImportError:
    HAS_HYBRID_MIDDLEWARE = False
    logger.warning("HybridMiddleware not available. Using legacy BLTEnhancedMiddleware.")

# Third-party dependencies
try:
    from pydantic import BaseModel, Field, validator
    from pydantic_settings import BaseSettings
except ImportError:
    print("Pydantic is not installed. Using basic config for BLTEnhancedMiddleware.")
    class BaseModel: pass
    class BaseSettings:
        def __init__(self, **kwargs): [setattr(self, k, v) for k,v in kwargs.items()]
    def Field(default, **kwargs): return default
    IS_PYDANTIC_AVAILABLE = False
else:
    IS_PYDANTIC_AVAILABLE = True

# --- VoxSigil Component Stubs/Interfaces (from previous script, simplified) ---
# Ensure these are your actual, robust implementations.
class SigilPatchEncoder: # Placeholder for actual BLT entropy/patch model
    def __init__(self, base_embedding_model=None, entropy_threshold=0.5):
        self.entropy_threshold = entropy_threshold
        logger.info(f"SigilPatchEncoder (BLT stub) initialized, threshold: {entropy_threshold}")
    
    def analyze_entropy(self, text: str) -> Tuple[Optional[List[str]], List[float]]:
        """
        Calculate entropy scores for text segments.
        This implementation actually calculates Shannon entropy and adds heuristic adjustments
        for certain text patterns, ensuring entropy values are never stuck at 0.0.
        """
        if not text: 
            return None, []
            
        # Split text into manageable patches
        patches = []
        entropy_scores = []
        
        # Determine patch size based on text length
        avg_patch_size = min(80, max(20, len(text) // 10))
        
        # Create patches
        for i in range(0, len(text), avg_patch_size):
            end = min(i + avg_patch_size, len(text))
            patch = text[i:end]
            patches.append(patch)
            
            # Shannon entropy calculation
            char_count = {}
            for char in patch:
                char_count[char] = char_count.get(char, 0) + 1
                
            patch_entropy = 0.0
            patch_len = len(patch)
            
            if patch_len > 0:
                for count in char_count.values():
                    freq = count / patch_len
                    patch_entropy -= freq * np.log2(freq)
                
                # Normalize to 0-1 range (typical text has entropy between 3.5-5)
                norm_entropy = min(1.0, patch_entropy / 8.0)
                
                # Apply heuristic adjustments based on content type
                if any(c in patch for c in ['<', '>', '{', '}', '[', ']']):
                    # Structured text tends to have lower entropy
                    norm_entropy = norm_entropy * 0.6
                elif any(keyword in patch.lower() for keyword in ['explain', 'describe', 'what is']):
                    # Natural language queries tend to have higher entropy
                    norm_entropy = norm_entropy * 1.2
                    
                entropy_scores.append(max(0.01, min(0.99, norm_entropy)))  # Never use 0.0
            else:
                entropy_scores.append(0.5)  # Default for empty patch
        
        # Ensure we have at least one patch and score
        if not patches:
            patches = [text]
        if not entropy_scores:
            entropy_scores = [0.5]
        
        # Log the calculated entropy
        avg_entropy = sum(entropy_scores) / len(entropy_scores)
        logger.debug(f"Calculated {len(patches)} patches with avg entropy {avg_entropy:.4f}")
        
        return patches, entropy_scores
        
    def encode(self, text:str) -> np.ndarray: # For BLT RAG embedding
        h = hashlib.sha256(f"blt_patch_emb_{text}".encode()).digest()
        return np.frombuffer(h, dtype=np.float32)[:128]

class VoxSigilRAG: # Placeholder for standard RAG
    def __init__(self):
        logger.info("Standard VoxSigilRAG (stub) initialized.")
        class MockEmb:
            def encode(self,t:str)->np.ndarray:
                h=hashlib.sha256(f"std_emb_{t}".encode()).digest()
                return np.frombuffer(h,dtype=np.float32)[:128]
        self.embedding_model = MockEmb()
    def create_rag_context(self, query:str, num_sigils:int=5, **kw) -> Tuple[str,List[Dict]]:
        emb = self.embedding_model.encode(query)
        s = [{"id":f"std_doc_{i}","content":f"Std content {i} for '{query[:20]}'", "score":np.random.rand()} for i in range(num_sigils)]
        return f"STD CONTEXT:\n"+"\n".join(d['content'] for d in s), s
    def retrieve_with_embedding(self, embedding: np.ndarray, query_text: str, num_sigils: int = 5, **kwargs) -> Tuple[str, List[Dict]]:
        # Assumes RAG can use a precomputed embedding for retrieval
        logger.debug(f"StandardRAG retrieving with precomputed embedding for '{query_text[:20]}'")
        # Simulate retrieval; in reality, this would use the embedding against an index
        s = [{"id": f"std_emb_doc_{i}", "content": f"Std (pre-emb) content {i} for '{query_text[:20]}'", "score": np.random.rand()} for i in range(num_sigils)]
        return f"STD CONTEXT (PRE-EMBEDDED):\n" + "\n".join(d['content'] for d in s), s

class PatchAwareValidator: # Stub
    def __init__(self, entropy_threshold=0.6): 
        logger.info("PatchAwareValidator (stub) initialized.")
        self.entropy_threshold = entropy_threshold
        
    def validate_schema(self, text:str) -> Tuple[bool, List[Dict]]: 
        return True, []
        
    def validate_structure(self, text:str) -> Tuple[bool, List[Dict]]:
        """
        Validate structure of a text using BLT-inspired patch analysis.
        
        Args:
            text: The text to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        # Simple implementation that always passes validation
        # In a real implementation, this would analyze the structure
        # of the text to ensure it meets expected patterns
        try:
            # Simple validation: check if the input is a non-empty string
            if not isinstance(text, str) or not text.strip():
                return False, [{"message": "Input must be a non-empty string"}]
                
            # Create a mock patch encoder for entropy calculation
            encoder = SigilPatchEncoder(entropy_threshold=self.entropy_threshold)
            _, entropy_scores = encoder.analyze_entropy(text)
            
            # Calculate average entropy
            avg_entropy = sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.5
            
            # Log for debugging
            logger.debug(f"PatchAwareValidator: Structure validation - avg_entropy={avg_entropy:.4f}")
            
            # Always return valid for now (placeholder implementation)
            return True, []
            
        except Exception as e:
            logger.warning(f"Error in validate_structure: {e}")
            return False, [{"message": f"Validation error: {str(e)}"}]

class PatchAwareCompressor: # Stub
    def __init__(self, entropy_threshold=0.5): logger.info("PatchAwareCompressor (stub) initialized.")
    def compress(self, text:str) -> Tuple[str, float]:
        if len(text) > 100 : return text[:len(text)//2] + "...", 0.5 # Dummy compression
        return text, 1.0
# --- END OF STUBS ---


# --- Configuration ---
class BLTMiddlewareConfig(BaseSettings):
    entropy_threshold: float = Field(0.35, description="Primary entropy threshold for routing.")
    blt_rag_weight: float = Field(0.7, description="Weight for BLT embedding in true hybrid scenarios.") # Note: different name than blt_hybrid_weight from problem
    entropy_router_fallback: str = Field("standard_rag", description="Fallback RAG if entropy fails ('blt_rag' or 'standard_rag').")
    cache_ttl_seconds: int = Field(360, description="TTL for RAG context cache.")
    log_level: str = Field("INFO", description="Logging level.")
    
    enable_patch_validation: bool = Field(True, description="Enable BLT-based patch validation of sigils/context.")
    enable_patch_compression: bool = Field(False, description="Enable BLT-based patch compression of context/query.")
    compression_max_size: Optional[int] = Field(2048, description="Max size for content before compression (if enabled).")
    
    # Semantic caching parameters
    enable_semantic_cache: bool = Field(True, description="Enable semantic cache for queries instead of exact string matching.")
    semantic_similarity_threshold: float = Field(0.85, description="Similarity threshold for semantic cache hits.")
    max_semantic_cache_size: int = Field(100, description="Maximum number of entries in the semantic cache.")
    
    # Recency boosting parameters
    enable_recency_boost: bool = Field(True, description="Enable boosting of relevance scores based on sigil recency.")
    recency_boost_factor: float = Field(0.2, description="Factor to boost recent sigils (0-1).")
    recency_max_days: int = Field(30, description="Number of days within which a sigil gets maximum recency boost.")

    # Parent middleware standard params (can be inherited or overridden)
    num_sigils: int = Field(5)
    min_score_threshold: float = Field(0.4)
    # ... other standard VoxSigilMiddleware params

    # class Config:
    #     env_prefix = "VOXSIGIL_BLT_"
    @validator('log_level')
    def set_log_level(cls, value): # Same as previous script
        numeric_level = getattr(logging, value.upper(), None)
        if not isinstance(numeric_level, int): raise ValueError(f"Invalid log level: {value}")
        logging.getLogger().setLevel(numeric_level)
        logging.getLogger("VoxSigilBLTMiddleware").setLevel(numeric_level)
        return value

# --- Parent Class STUB (from above) ---
class VoxSigilMiddleware:
    def __init__(self, voxsigil_rag_instance: Optional[VoxSigilRAG] = None, num_sigils: int = 5, min_score_threshold:float=0.4, **kwargs):
        self._standard_rag_instance = voxsigil_rag_instance if voxsigil_rag_instance else VoxSigilRAG()
        self.num_sigils = num_sigils
        self.min_score_threshold = min_score_threshold
        logger.info(f"Base VoxSigilMiddleware initialized. Num sigils: {self.num_sigils}")
    def preprocess_request(self, request: Dict[str, Any]) -> Dict[str, Any]: return request
    def _extract_query_from_request(self, request: Dict[str, Any]) -> Optional[str]:
        # ... (implementation from above) ...
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages: return None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        last_message = messages[-1]
        if isinstance(last_message, dict) and isinstance(last_message.get("content"), str):
            return last_message["content"]
        return None
    def _enhance_request_with_context(self, request: Dict[str, Any], context: str, context_source: str = "Hybrid") -> Dict[str, Any]:
        messages = request.get("messages", [])
        if not context or not messages: return request
        enhanced_messages = [msg.copy() for msg in messages]
        system_message_content = (
            f"Source: {context_source}. Use the following retrieved VoxSigil context:\n\n"
            f"--- VOXSIGIL CONTEXT START ---\n{context}\n--- VOXSIGIL CONTEXT END ---"
        )
        # Find and prepend to system message, or insert new one
        prepended = False
        for i, msg in enumerate(enhanced_messages):
            if msg.get("role") == "system":
                original_content = msg.get("content", "")
                enhanced_messages[i]["content"] = f"{system_message_content}\n\n{original_content}"
                prepended = True
                break
        if not prepended:
            enhanced_messages.insert(0, {"role": "system", "content": system_message_content})
        request["messages"] = enhanced_messages
        return request
    def wrap_llm_api(self, llm_api_call: Callable[..., Any]) -> Callable[..., Any]: # Identical to stub
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.debug("Base VoxSigilMiddleware: Wrapping LLM API call.")
            return llm_api_call(*args, **kwargs)
        return wrapper

# Configure logging globally for the module
logger = logging.getLogger("VoxSigilBLTMiddleware")
# (log level will be set by config)

# --- Core Hybrid Logic Components (adapted from previous production script) ---
class BLTEntropyRouter: # Feature 1 & 2 & 8
    def __init__(self, config: BLTMiddlewareConfig):
        self.config = config
        # For BLT-Enhanced, SigilPatchEncoder is primary for entropy
        self.patch_encoder = SigilPatchEncoder(entropy_threshold=config.entropy_threshold)
        logger.info(f"BLTEntropyRouter initialized: threshold={config.entropy_threshold}, fallback_rag={config.entropy_router_fallback}")
        
    def route(self, text: str) -> Tuple[str, Optional[List[str]], List[float]]:
        if not text:
            logger.warning("Empty text for routing. Using fallback RAG.")
            return self.config.entropy_router_fallback, None, [0.5]

        try:
            patches, entropy_scores = self.patch_encoder.analyze_entropy(text)
            if not entropy_scores: # Heuristic fallback if analyze_entropy gives no scores
                logger.warning(f"No entropy scores from SigilPatchEncoder for: '{text[:30]}...'. Applying heuristic.")
                avg_entropy = 0.15 if any(c in text for c in ['<','>','{','}']) else 0.75
                entropy_scores = [avg_entropy]
                patches = patches or [text]
            
            avg_entropy = sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.5
            
            # Enhanced logging for entropy routing
            route_decision = "blt_rag" if avg_entropy < self.config.entropy_threshold else "standard_rag"
            logger.info(f"ENTROPY ROUTING: text='{text[:30]}...', avg_entropy={avg_entropy:.4f}, threshold={self.config.entropy_threshold:.4f}, route={route_decision}")
            
            # Output entropy scores for debugging
            if len(entropy_scores) <= 10:
                logger.debug(f"Entropy scores: {[round(e, 4) for e in entropy_scores]}")
            else:
                logger.debug(f"Entropy scores (sample of 10): {[round(e, 4) for e in entropy_scores[:10]]}, ... (total: {len(entropy_scores)})")
            
            # Route to "blt_rag" for low entropy, "standard_rag" for high
            return route_decision, patches, entropy_scores
        except Exception as e:
            logger.error(f"Entropy routing failed: {e}. Using fallback_rag: {self.config.entropy_router_fallback}", exc_info=True)
            return self.config.entropy_router_fallback, None, [0.5]

# Feature 9 (Conceptual)
class BLTDynamicExecutionBudgeter:
    def __init__(self, base_budget: float = 1.0, entropy_multiplier: float = 1.2):
        self.base_budget = base_budget; self.entropy_multiplier = entropy_multiplier
    def allocate_budget(self, method: str, avg_entropy: float, length: int) -> float:
        budget = self.base_budget * ( (1.0 - 0.4 * avg_entropy) if "blt" in method else (1.0 + self.entropy_multiplier * avg_entropy) )
        budget *= max(0.5, min(2.0, length / 400.0))
        return budget

# --- Main BLT-Enhanced Middleware ---
class BLTEnhancedMiddleware(VoxSigilMiddleware): # Inherits from VoxSigilMiddleware
    """
    BLT-enhanced middleware integrating hybrid RAG routing, caching, and BLT features.
    
    DEPRECATED: This class is maintained for backward compatibility but is now replaced
    by the more powerful HybridMiddleware class from hybrid_blt.py.
    
    For new code, please use:
    from VoxSigilRag.hybrid_blt import HybridMiddleware
    """
    def __init__(self, 
                 config: Optional[BLTMiddlewareConfig] = None,
                 voxsigil_rag_instance: Optional[VoxSigilRAG] = None, # Standard RAG
                 # Other parent params can be passed or come from config
                 **kwargs): # Allows passing parent params directly
                 
        # Emit deprecation warning
        logger.warning(
            "BLTEnhancedMiddleware is deprecated and will be removed in future versions. "
            "Please use HybridMiddleware from VoxSigilRag.hybrid_blt instead."
        )

        # Feature 10: Configuration
        self.config = config if config else BLTMiddlewareConfig(**kwargs) # Load/use config

        # Initialize parent with relevant config/params
        super().__init__(
            voxsigil_rag_instance=voxsigil_rag_instance, # This is the "standard_rag"
            num_sigils=self.config.num_sigils,
            min_score_threshold=self.config.min_score_threshold,
            # Pass other relevant parent params from self.config or kwargs
            **{k:v for k,v in kwargs.items() if k not in BLTMiddlewareConfig.__fields__}
        )
        
        self.router = BLTEntropyRouter(self.config)
        self.budgeter = BLTDynamicExecutionBudgeter()

        # Feature 5: Lazy Init for BLT-specific RAG components (or eager if lightweight stubs)
        # For BLT RAG, we use SigilPatchEncoder for embeddings. The actual retrieval
        # might still use an underlying vector store, potentially shared or specialized.
        # We assume self.patch_encoder IS the core of the BLT embedding generation.
        self._patch_encoder_instance: Optional[SigilPatchEncoder] = None # For BLT embeddings

        # BLT-specific components from the original BLTEnhancedMiddleware
        self.patch_validator = PatchAwareValidator(entropy_threshold=self.config.entropy_threshold + 0.1)
        self.patch_compressor = PatchAwareCompressor(entropy_threshold=self.config.entropy_threshold)

        # Feature 6: Context Caching
        self._rag_context_cache: Dict[str, Tuple[str, List[Dict[str, Any]], str, float]] = {} # key: (context, sigils, route, timestamp)
        
        # Feature 11: Semantic Caching
        self.semantic_cache = None
        if self.config.enable_semantic_cache:
            self.semantic_cache = SemanticCacheManager(
                similarity_threshold=self.config.semantic_similarity_threshold,
                ttl_seconds=self.config.cache_ttl_seconds,
                max_cache_size=self.config.max_semantic_cache_size
            )
            # We'll set the embedding function later once we have the encoder ready
        self._request_counter = 0
        self._processing_times = []

        logger.info(f"BLTEnhancedMiddleware initialized with Hybrid capabilities. Cache TTL: {self.config.cache_ttl_seconds}s.")
        if self.config.enable_semantic_cache:
            logger.info(f"Semantic caching enabled with similarity threshold: {self.config.semantic_similarity_threshold}")
        if self.config.enable_recency_boost:
            logger.info(f"Recency boosting enabled with factor: {self.config.recency_boost_factor}, max days: {self.config.recency_max_days}")
            
    @property
    def blt_patch_encoder(self) -> SigilPatchEncoder: # Lazy init for BLT encoder if it were heavy
        if self._patch_encoder_instance is None:
            logger.info("Lazy initializing SigilPatchEncoder for BLTEnhancedMiddleware...")
            # Base model could come from standard RAG if needed for patch encoder init
            base_model_for_patch_encoder = self._standard_rag_instance.embedding_model \
                if hasattr(self._standard_rag_instance, 'embedding_model') else None
            self._patch_encoder_instance = SigilPatchEncoder(
                base_embedding_model=base_model_for_patch_encoder,
                entropy_threshold=self.config.entropy_threshold
            )
            
            # Set up the embedding function for the semantic cache if enabled
            if self.config.enable_semantic_cache and self.semantic_cache:
                self.semantic_cache.set_embedding_function(self._patch_encoder_instance.encode)
                logger.info("Set up embedding function for semantic cache")
        
        return self._patch_encoder_instance

    def _get_cache_key(self, query: str) -> str: # Identical to previous
        norm_q = ' '.join(query.lower().strip().split())
        return hashlib.sha256(norm_q.encode()).hexdigest() if len(norm_q) > 256 else norm_q

    def _clean_expired_cache_entries(self): # Identical to previous
        current_time = time.monotonic()
        expired = [k for k, (*_, ts) in self._rag_context_cache.items() if current_time - ts > self.config.cache_ttl_seconds]
        for k in expired: del self._rag_context_cache[k]
        if expired: logger.info(f"Cleaned {len(expired)} expired RAG cache entries.")    # Feature 3 & 4: Differentiated RAG processing and Fallback
    def _get_hybrid_rag_context(self, query: str) -> Tuple[str, List[Dict[str, Any]], str, float]:
        """Core hybrid RAG logic: route, retrieve, handle errors."""
        # First check semantic cache if enabled
        if self.config.enable_semantic_cache and self.semantic_cache:
            semantic_cache_result = self.semantic_cache.get(query)
            if semantic_cache_result:
                cached_data, similarity = semantic_cache_result
                context_str, sigils, route_method, avg_entropy = cached_data
                logger.info(f"Semantic cache hit with similarity {similarity:.4f} for query: '{query[:30]}...'")
                
                # Apply recency boosting even to cached results since time has passed
                if self.config.enable_recency_boost:
                    sigils = self._apply_recency_boost(sigils)
                    
                    # If we boosted the sigils, we need to update the context string
                    if sigils:
                        # Simplified re-creation of context - in a real system, might need more logic
                        context_str = "\n".join(s.get('content', '') for s in sigils if s.get('content'))
                
                return context_str, sigils, f"{route_method}_semantic_cache", avg_entropy
        
        route_decision, _, entropy_scores = self.router.route(query)
        avg_entropy = sum(entropy_scores)/len(entropy_scores) if entropy_scores else 0.5
        
        context_str, sigils, actual_method_used = "", [], route_decision

        try:
            if route_decision == "blt_rag": # Low entropy path
                logger.debug(f"Query '{query[:30]}...' routed to BLT RAG (entropy: {avg_entropy:.2f})")
                # Use SigilPatchEncoder for embedding, then retrieve.
                # This assumes your RAG system can take a BLT-style embedding
                # or that SigilPatchEncoder produces embeddings compatible with your standard RAG's index.
                # For simplicity, let's assume the patch encoder gives the vector.
                blt_embedding = self.blt_patch_encoder.encode(query)
                # Now, how does BLT RAG retrieve? Does it use the standard RAG's index with this embedding?
                # Or a separate BLT-specific index/retrieval?
                # Assuming it can use the standard RAG store with a different embedding type:
                context_str, sigils = self._standard_rag_instance.retrieve_with_embedding(
                                            blt_embedding, query, num_sigils=self.config.num_sigils) # Pass query for context in sigils
                actual_method_used = "blt_rag_on_std_index" # More descriptive method

            elif route_decision == "standard_rag": # High entropy or fallback path
                logger.debug(f"Query '{query[:30]}...' routed to Standard RAG (entropy: {avg_entropy:.2f})")
                context_str, sigils = self._standard_rag_instance.create_rag_context(
                                            query, num_sigils=self.config.num_sigils)
            else: # Should not happen if router is correct
                logger.error(f"Unknown RAG route: {route_decision}. Defaulting to standard.")
                context_str, sigils = self._standard_rag_instance.create_rag_context(
                                            query, num_sigils=self.config.num_sigils)
                actual_method_used = "standard_rag_fallback_unknown_route"
        
        except Exception as e:
            logger.error(f"Error in RAG path '{actual_method_used}' for query '{query[:30]}...': {e}", exc_info=True)
            # Intelligent Fallback: try the other RAG path
            if actual_method_used.startswith("blt"):
                logger.warning("BLT RAG failed. Attempting fallback to Standard RAG.")
                try:
                    context_str, sigils = self._standard_rag_instance.create_rag_context(query, num_sigils=self.config.num_sigils)
                    actual_method_used = "standard_rag_from_blt_failure"
                except Exception as fallback_e:
                    logger.critical(f"Standard RAG fallback also failed: {fallback_e}", exc_info=True)
                    context_str, sigils = "", [] # Critical failure
            else: # Standard RAG failed
                logger.warning("Standard RAG failed. Attempting fallback to BLT RAG (if applicable).")
                try:
                    blt_embedding = self.blt_patch_encoder.encode(query)
                    context_str, sigils = self._standard_rag_instance.retrieve_with_embedding(blt_embedding, query, num_sigils=self.config.num_sigils)
                    actual_method_used = "blt_rag_from_std_failure"
                except Exception as fallback_e:
                    logger.critical(f"BLT RAG fallback also failed: {fallback_e}", exc_info=True)
                    context_str, sigils = "", [] # Critical failure
        
        # Apply recency boosting
        if self.config.enable_recency_boost and sigils:
            sigils = self._apply_recency_boost(sigils)
            
            # Update context string with boosted sigils
            # Simplified re-creation - in a real system, might need more logic
            if context_str and sigils:
                context_str = "\n".join(s.get('content', '') for s in sigils if s.get('content'))
        
        # Store in semantic cache if enabled
        if self.config.enable_semantic_cache and self.semantic_cache and query and (context_str or sigils):
            cache_data = (context_str, sigils, actual_method_used, avg_entropy)
            self.semantic_cache.add(query, cache_data)
            logger.debug(f"Added to semantic cache: '{query[:30]}...'")
        
        return context_str, sigils, actual_method_used, avg_entropy
        
    def _apply_blt_specific_enhancements(self, request: Dict[str, Any], context_str: str, sigils: List[Dict]) -> Tuple[Dict[str,Any], str]:
        """Applies patch validation and compression if enabled."""
        enhanced_request = request.copy()
        final_context = context_str

        # 1. Patch Validation (e.g., on retrieved sigil content)
        if self.config.enable_patch_validation and sigils:
            valid_sigils = []
            for sigil in sigils:
                content_to_validate = str(sigil.get("content", "")) # Or more specific fields
                is_valid, issues = self.patch_validator.validate_schema(content_to_validate)
                if is_valid:
                    valid_sigils.append(sigil)
                else:
                    logger.warning(f"Sigil ID {sigil.get('id', 'N/A')} failed patch validation: {issues}")
            
            if len(valid_sigils) != len(sigils): # Rebuild context if some sigils were invalid
                final_context = "\n".join(s['content'] for s in valid_sigils if s.get('content'))
                logger.info(f"Rebuilt context after patch validation: {len(valid_sigils)}/{len(sigils)} sigils valid.")
                # Update sigils in request metadata if it's stored there.

        # 2. Patch Compression (e.g., on final context or user query)
        if self.config.enable_patch_compression:
            # Compress user query
            user_query = self._extract_query_from_request(enhanced_request)
            if user_query and (not self.config.compression_max_size or len(user_query) > self.config.compression_max_size):
                compressed_query, ratio = self.patch_compressor.compress(user_query)
                if ratio < 0.95: # Only replace if significant compression
                    logger.debug(f"Compressing user query (ratio: {ratio:.2f})")
                    # Update query in messages
                    for msg in enhanced_request.get("messages", []):
                        if msg.get("role") == "user" and msg.get("content") == user_query:
                            msg["content"] = compressed_query
                            break
            
            # Compress RAG context
            if final_context and (not self.config.compression_max_size or len(final_context) > self.config.compression_max_size):
                final_context, ratio = self.patch_compressor.compress(final_context)
                logger.debug(f"Compressed RAG context (ratio: {ratio:.2f})")
        
        return enhanced_request, final_context

    def _apply_recency_boost(self, sigils: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply recency boosting to sigil relevance scores.
        
        Args:
            sigils: List of sigil dictionaries with scores
            
        Returns:
            List of sigils with boosted scores
        """
        if not self.config.enable_recency_boost or not sigils:
            return sigils
            
        now = datetime.datetime.now()
        max_days = self.config.recency_max_days
        boost_factor = self.config.recency_boost_factor
        
        boosted_sigils = []
        
        for sigil in sigils:
            # Copy the sigil to avoid modifying the original
            boosted_sigil = sigil.copy()
            
            # Check if the sigil has a timestamp
            timestamp = None
            if 'timestamp' in sigil:
                timestamp = sigil['timestamp']
            elif 'last_updated' in sigil:
                timestamp = sigil['last_updated']
            elif 'created_at' in sigil:
                timestamp = sigil['created_at']
                
            # If no timestamp, use current time or skip boosting
            if timestamp is None:
                boosted_sigils.append(boosted_sigil)
                continue
                
            # Convert timestamp to datetime if it's a string
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    try:
                        # Try a few common formats if ISO format fails
                        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%m/%d/%Y']:
                            try:
                                timestamp = datetime.datetime.strptime(timestamp, fmt)
                                break
                            except ValueError:
                                continue
                    except Exception:
                        # If all parsing fails, skip boosting this sigil
                        boosted_sigils.append(boosted_sigil)
                        continue
            
            # Calculate days since the timestamp
            if isinstance(timestamp, datetime.datetime):
                days_diff = (now - timestamp).days
                
                # Apply recency boost
                recency_ratio = max(0, 1 - (days_diff / max_days)) if days_diff <= max_days else 0
                original_score = boosted_sigil.get('score', 0.5)
                
                # Boost formula: original_score + (1 - original_score) * boost_factor * recency_ratio
                # This ensures that:
                # 1. Higher original scores still get relatively smaller boosts
                # 2. More recent items get larger boosts
                # 3. Score never exceeds 1.0
                boosted_score = original_score + (1 - original_score) * boost_factor * recency_ratio
                boosted_sigil['score'] = min(1.0, boosted_score)
                
                # Add metadata about boosting
                boosted_sigil['recency_boost_applied'] = True
                boosted_sigil['original_score'] = original_score
                boosted_sigil['recency_boost'] = boosted_score - original_score
            
            boosted_sigils.append(boosted_sigil)
            
        # Re-sort by boosted scores
        boosted_sigils.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return boosted_sigils

    # --- Overriding parent methods ---
    def preprocess_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for enhancing a request. Overrides parent.
        Implements hybrid RAG routing, caching, and BLT-specific enhancements.
        """
        self._request_counter += 1
        start_time_total = time.monotonic()

        query = self._extract_query_from_request(request)
        if not query:
            logger.warning("No query extractable. Passing request to super or returning as is.")
            return super().preprocess_request(request) # Or just request

        # Clean cache periodically
        if self._request_counter % 25 == 0: 
            self._clean_expired_cache_entries()
            # Also clean semantic cache if enabled
            if self.config.enable_semantic_cache and self.semantic_cache:
                self.semantic_cache.clean_expired()

        # Check if we're using standard exact-match caching or semantic caching
        current_monotonic_time = time.monotonic()
        cache_hit = False
        semantic_cache_hit = False
        semantic_similarity = 0.0
        
        context_str, sigils, route_method, avg_entropy = "", [], "unknown", 0.5
        
        # We're now using semantic caching as the primary cache mechanism if enabled
        # If semantic caching is disabled, or if we get a semantic cache miss, fall back to the exact string cache
        
        # Skip caching altogether if query is too short
        if len(query) >= 3:
            # Try semantic cache first
            if self.config.enable_semantic_cache and self.semantic_cache:
                semantic_result = self.semantic_cache.get(query)
                if semantic_result:
                    cached_data, similarity = semantic_result
                    context_str, sigils, route_method, avg_entropy = cached_data
                    semantic_cache_hit = True
                    semantic_similarity = similarity
                    logger.info(f"Semantic cache HIT for query (similarity: {similarity:.4f}). Using cached RAG context. Route: {route_method}")
                    cache_hit = True
                else:
                    logger.info(f"Semantic cache MISS for query: '{query[:30]}...'")
            
            # If semantic cache missed or is disabled, try exact string cache
            if not cache_hit:
                cache_key = self._get_cache_key(query)
                cached_entry = self._rag_context_cache.get(cache_key)
                
                if cached_entry:
                    _ctx, _sgls, _route, _ts = cached_entry
                    if current_monotonic_time - _ts <= self.config.cache_ttl_seconds:
                        logger.info(f"Exact string cache HIT for query (key: {cache_key[:30]}...). Using cached RAG context. Route: {_route}")
                        context_str, sigils, route_method = _ctx, _sgls, _route
                        self._rag_context_cache[cache_key] = (_ctx, _sgls, _route, current_monotonic_time) # Refresh timestamp
                        cache_hit = True
                    else:
                        logger.info(f"Exact string cache STALE for query (key: {cache_key[:30]}...). Re-computing.")
                        del self._rag_context_cache[cache_key]
        
        if not cache_hit:
            logger.info(f"All caches MISS for query: '{query[:30]}...' - Processing with Hybrid RAG.")
            context_str, sigils, route_method, avg_entropy = self._get_hybrid_rag_context(query)
            
            # Store in exact match cache if we got results
            if context_str or sigils: 
                cache_key = self._get_cache_key(query)
                self._rag_context_cache[cache_key] = (context_str, sigils, route_method, current_monotonic_time)
                
                # Note: Semantic caching is handled directly inside _get_hybrid_rag_context

        # Apply BLT-specifics like validation/compression AFTER RAG retrieval
        enhanced_request, final_context_str = self._apply_blt_specific_enhancements(request, context_str, sigils)
        
        # Inject context into the request (using parent's helper or similar)
        cache_source = "Semantic" if semantic_cache_hit else "Exact" if cache_hit else "None"
        final_enhanced_request = self._enhance_request_with_context(
            enhanced_request, 
            final_context_str, 
            context_source=f"Hybrid-{route_method}-Cache{cache_source}"
        )
        
        # --- Metadata & Budgeting ---
        processing_time = time.monotonic() - start_time_total
        self._processing_times.append(processing_time)
        budget = self.budgeter.allocate_budget(route_method, avg_entropy, len(query))
        
        log_meta = {
            "query_preview": query[:40]+"...", 
            "route": route_method, 
            "cache": cache_source,
            "semantic_similarity": round(semantic_similarity, 4) if semantic_cache_hit else 0.0,
            "ctx_len": len(final_context_str), 
            "sigils": len(sigils), 
            "budget": round(budget, 2),
            "time_ms": round(processing_time*1000, 2), 
            "avg_entropy": round(avg_entropy, 3)
        }
        logger.info(f"BLT Middleware processed: {log_meta}")
        final_enhanced_request.setdefault("voxsigil_blt_metadata", {}).update(log_meta)

        return final_enhanced_request

    # wrap_llm_api could be overridden if BLT needs to inspect/modify LLM *responses*
    # or do something more complex than parent's wrapper with the request.
    # For now, let's assume parent's wrap_llm_api is sufficient or BLT mods are in preprocess.
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the BLTEnhancedMiddleware, including semantic cache and recency boosting.
        
        Returns:
            Dictionary of statistics
        """
        avg_time = sum(self._processing_times) / len(self._processing_times) if self._processing_times else 0
        stats = {
            "total_requests": self._request_counter, 
            "exact_cache_size": len(self._rag_context_cache),
            "avg_processing_time_s": round(avg_time, 4),
        }
        
        # Add semantic cache stats if available
        if self.config.enable_semantic_cache and self.semantic_cache:
            stats.update({
                "semantic_cache_enabled": True,
                "semantic_cache_stats": self.semantic_cache.get_stats()
            })
        else:
            stats["semantic_cache_enabled"] = False
            
        # Add recency boosting stats
        stats["recency_boosting_enabled"] = self.config.enable_recency_boost
        if self.config.enable_recency_boost:
            stats["recency_boost_factor"] = self.config.recency_boost_factor
            stats["recency_max_days"] = self.config.recency_max_days
            
        return stats

# --- Utility Function for True Hybrid Embedding (Feature 7) ---
# This is a helper, can be called by RAG components if "true hybrid" blend is needed.
def compute_true_hybrid_embedding(
        text: str,
        standard_encoder_model: Any, # e.g., self._standard_rag_instance.embedding_model
        blt_patch_encoder_model: SigilPatchEncoder, # e.g., self.blt_patch_encoder
        blt_weight: float,
        target_dim: Optional[int] = None # Optional: to ensure consistent dimensionality
    ) -> Optional[np.ndarray]:
    logger.debug(f"Computing true hybrid embedding for: '{text[:30]}...' with BLT weight: {blt_weight}")
    try:
        std_emb = standard_encoder_model.encode(text)
        blt_emb = blt_patch_encoder_model.encode(text)        # Ensure dimensions match if target_dim is provided, or if they differ.
        # This is a placeholder for actual dimension alignment logic (padding/truncation/projection)
        if target_dim:
            std_emb = std_emb[:target_dim] if len(std_emb) > target_dim else np.pad(std_emb, (0, target_dim - len(std_emb)))
            blt_emb = blt_emb[:target_dim] if len(blt_emb) > target_dim else np.pad(blt_emb, (0, target_dim - len(blt_emb)))
        elif std_emb.shape != blt_emb.shape:
            logger.warning(f"Std emb shape {std_emb.shape} != BLT emb shape {blt_emb.shape}. Attempting resize to min dim.")
            min_d = min(std_emb.shape[0], blt_emb.shape[0])
            std_emb, blt_emb = std_emb[:min_d], blt_emb[:min_d]
            
        std_norm = std_emb / (np.linalg.norm(std_emb) + 1e-9)
        blt_norm = blt_emb / (np.linalg.norm(blt_emb) + 1e-9)
        
        # Only merge vectors if the weight is not at the extremes (0 or 1)
        # This avoids unnecessary computation when using only one embedding type
        if blt_weight <= 0.01:
            logger.info("Using 100% standard embedding (no hybrid merge)")
            return std_norm  # Just use standard embedding
        elif blt_weight >= 0.99:
            logger.info("Using 100% BLT embedding (no hybrid merge)")
            return blt_norm  # Just use BLT embedding
        else:
            logger.info(f"Computing hybrid embedding with BLT weight: {blt_weight:.2f}")
            hybrid_emb = (blt_weight * blt_norm) + ((1 - blt_weight) * std_norm)
            return hybrid_emb / (np.linalg.norm(hybrid_emb) + 1e-9)
    except Exception as e:
        logger.error(f"Failed to compute true hybrid embedding: {e}", exc_info=True)
        return None


class ByteLatentTransformerEncoder:
    """
    Implementation of the Byte Latent Transformer Encoder for handling embeddings.
    This encoder is designed for the BLT (Byte Latent Transformer) architecture.
    """
    def __init__(self, base_embedding_model=None, patch_size=64, max_patches=16):
        self.base_embedding_model = base_embedding_model
        self.patch_size = patch_size
        self.max_patches = max_patches
        logger.info(f"ByteLatentTransformerEncoder initialized with patch_size={patch_size}, max_patches={max_patches}")
        
    def validate_input(self, text: Any) -> str:
        """
        Validate and normalize input text for BLT processing.
        
        Args:
            text: The input to validate (should be string)
            
        Returns:
            Normalized string version of input
            
        Raises:
            TypeError: If input cannot be converted to string
        """
        if text is None:
            logger.warning("Received None input in validate_input")
            return ""
            
        if isinstance(text, str):
            return text
            
        # Handle common encodable types
        if isinstance(text, (int, float, bool)):
            return str(text)
            
        # Handle bytes
        if isinstance(text, bytes):
            try:
                return text.decode('utf-8', errors='replace')
            except Exception as e:
                logger.warning(f"Error decoding bytes: {e}")
                return str(text)
                
        # Handle lists, dicts, etc.
        try:
            return str(text)
        except Exception as e:
            logger.error(f"Failed to convert input to string: {e}")
            raise TypeError(f"Input must be convertible to string, got {type(text)}")
            
    def create_patches(self, text: str) -> List[Any]:
        """
        Create byte-oriented patches for the input text using BLT principles.
        
        Args:
            text: The input text to create patches for
            
        Returns:
            List of patch objects with entropy attributes
        """
        # Validate and normalize input
        try:
            validated_text = self.validate_input(text)
        except TypeError as e:
            logger.error(f"Invalid input to create_patches: {e}")
            return []
            
        if not validated_text:
            return []
            
        # Create a simple patch representation with entropy estimation
        class Patch:
            def __init__(self, content, start_pos, end_pos, entropy):
                self.content = content
                self.start_pos = start_pos
                self.end_pos = end_pos
                self.entropy = entropy
                
        patches = []
        text_bytes = validated_text.encode('utf-8')
        text_len = len(text_bytes)
        
        # Create patches based on patch_size
        for i in range(0, text_len, self.patch_size):
            end_pos = min(i + self.patch_size, text_len)
            chunk = text_bytes[i:end_pos]
            
            # Simple entropy calculation (Shannon entropy)
            if chunk:
                frequencies = {}
                for byte in chunk:
                    if byte not in frequencies:
                        frequencies[byte] = 0
                    frequencies[byte] += 1
                
                entropy = 0
                for count in frequencies.values():
                    probability = count / len(chunk)
                    entropy -= probability * np.log2(probability)
            else:
                entropy = 0
                
            # Create patch
            chunk_text = chunk.decode('utf-8', errors='replace')
            patch = Patch(chunk_text, i, end_pos, entropy)
            patches.append(patch)
            
            # Limit number of patches
            if len(patches) >= self.max_patches:
                break
                
        return patches
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text into a vector embedding using BLT principles.
        Always returns a 128-dimensional vector for consistency.
        """
        logger.debug(f"BLTEncoder: Encoding text: '{text[:50]}...'")
        
        if not text:
            # Return zero embedding for empty text
            return np.zeros(128, dtype=np.float32)
            
        # If we have a base embedding model, use it
        if self.base_embedding_model is not None:
            try:
                embedding = self.base_embedding_model.encode(text)
                # Ensure consistent dimension of 128
                if len(embedding) < 128:
                    padded = np.zeros(128, dtype=np.float32)
                    padded[:len(embedding)] = embedding
                    return padded
                return embedding[:128]  # Truncate if longer than 128
            except Exception as e:
                logger.warning(f"Error using base embedding model: {e}. Falling back to default BLT encoding.")
        
        # Default BLT encoding using a hash-based approach
        # In a real implementation, this would use a more sophisticated 
        # byte-level transformation, potentially with learned weights
        h = hashlib.sha256(f"blt_{text}".encode()).digest()
        # Ensure we get exactly 128 floats by padding or truncating
        raw_result = np.frombuffer(h, dtype=np.float32)
        
        # Create a zero array of correct size
        result = np.zeros(128, dtype=np.float32)
        
        # Copy available data, which might be less than 128 elements
        result[:min(len(raw_result), 128)] = raw_result[:min(len(raw_result), 128)]
        
        # Normalize the vector
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
            
        return result
        
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts into vector embeddings.
        Ensures all output vectors have consistent dimensions.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            NumPy array with shape (len(texts), 128)
        """        # Create a properly sized output array
        batch_size = len(texts)
        result = np.zeros((batch_size, 128), dtype=np.float32)
        
        # Fill the array one encoding at a time
        for i, text in enumerate(texts):
            result[i] = self.encode(text)
            
        return result


# --- Main Example Usage ---
if __name__ == "__main__":
    print("=" * 80 + "\n🚀 VoxSigil BLT-Enhanced Hybrid Middleware Test 🚀\n" + "=" * 80)

    # Test Config
    test_conf_obj = BLTMiddlewareConfig(
        log_level="DEBUG", 
        entropy_threshold=0.4,
        cache_ttl_seconds=10, # Short TTL for testing
        enable_patch_compression=True, # Test compression
        compression_max_size=100 # Small for testing compression
    )
    logger.info(f"Test configuration: {test_conf_obj.model_dump_json(indent=2) if IS_PYDANTIC_AVAILABLE else vars(test_conf_obj)}")

    # Initialize Middleware
    # Standard RAG (stub) will be created by default if not passed
    blt_middleware = BLTEnhancedMiddleware(config=test_conf_obj)

    test_requests_data = [
        {"messages": [{"role": "user", "content": "Explain symbolic reasoning in AI."}]}, # High entropy expected
        {"messages": [{"role": "user", "content": "<query type='sigils'><filter field='id'>S123</filter></query>"}]}, # Low entropy
        {"messages": [{"role": "user", "content": "Tell me about {code_block: 'function foo() { return 1; }'}"}]}, # Mixed
        {"messages": [{"role": "user", "content": "Explain symbolic reasoning in AI."}]}, # Test cache
        {"messages": [{"role": "user", "content": "A very long query" * 20 + "that should trigger compression if enabled and above threshold."}]}, # Test compression
    ]

    for i, req_data_item in enumerate(test_requests_data):
        print(f"\n--- Processing Request {i+1} ---")
        original_query = blt_middleware._extract_query_from_request(req_data_item)
        print(f"Original Query Preview: '{original_query[:60]}...'")
        
        processed_req = blt_middleware.preprocess_request(req_data_item.copy())
        
        final_query = blt_middleware._extract_query_from_request(processed_req) # Query might be compressed
        print(f"Final Query Preview (after BLT): '{final_query[:60]}...'")
        if processed_req["messages"][0].get("role") == "system":
            print(f"System Context Preview: {processed_req['messages'][0]['content'][:150]}...")
        metadata = processed_req.get("voxsigil_blt_metadata", {})
        print(f"Metadata: Route='{metadata.get('route')}', Cache='{metadata.get('cache')}', Time='{metadata.get('time_ms')}ms'")

    print(f"\n--- Middleware Stats after {len(test_requests_data)} requests ---")
    print(blt_middleware.get_stats())

    # Test True Hybrid Embedding Utility
    print("\n\n--- Testing True Hybrid Embedding Utility ---")
    standard_rag_for_util = VoxSigilRAG() # Get a standard RAG instance
    patch_encoder_for_util = SigilPatchEncoder(entropy_threshold=0.4) # Get a patch encoder
    
    hybrid_emb_test = compute_true_hybrid_embedding(
        "Test query for hybrid embedding.",
        standard_encoder_model=standard_rag_for_util.embedding_model,
        blt_patch_encoder_model=patch_encoder_for_util,
        blt_weight=test_conf_obj.blt_rag_weight, # from config
        target_dim=128
    )
    if hybrid_emb_test is not None:
        print(f"True Hybrid Embedding computed, shape: {hybrid_emb_test.shape}, first 3: {hybrid_emb_test[:3]}")
    else:
        print("True Hybrid Embedding computation FAILED.")
        
    print("\n" + "=" * 80 + "\n✅ BLT-Enhanced Hybrid Middleware Test Complete ✅\n" + "=" * 80)