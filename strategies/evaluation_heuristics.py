# voxsigil_supervisor/evaluation_heuristics.py
"""
Provides evaluation and quality assessment functionality for the VoxSigil Supervisor.
Connects to the existing VoxSigil Evaluator component to assess responses.
"""
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import re
from pathlib import Path
import sys

# Setup the logger
logger_evaluation = logging.getLogger("VoxSigilSupervisor.evaluation")

# Simple evaluator that doesn't depend on external components
class SimpleEvaluator:
    """
    A simplified evaluator that doesn't depend on external components.
    This class provides basic evaluation functionality for responses.
    """
    
    def __init__(self):
        """Initialize the SimpleEvaluator."""
        self.logger = logger_evaluation
        self.logger.info("Initialized SimpleEvaluator")
        
    def evaluate(self, query: str, response: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of a response based on simple heuristics.
        
        Args:
            query: The original user query
            response: The generated response (can be a string or a tuple where the first element is the response text)
            context: Optional context used for generation
            
        Returns:
            A dictionary with evaluation metrics
        """
        self.logger.info("Evaluating response using simple heuristics")
        
        # Handle case where response is a tuple (e.g., from process_query return value)
        if isinstance(response, tuple) and len(response) > 0:
            response_text = response[0]
        else:
            response_text = response
        
        # Ensure response_text is a string
        if not isinstance(response_text, str):
            self.logger.warning(f"Expected string for response but got {type(response_text)}. Converting to string.")
            try:
                response_text = str(response_text)
            except Exception as e:
                self.logger.error(f"Failed to convert {type(response_text)} to string: {e}")
                response_text = ""
        
        # Calculate simple metrics
        response_length = len(response_text)
        query_terms = set(self._extract_keywords(query))
        response_terms = set(self._extract_keywords(response_text))
        
        # Check term overlap
        if query_terms:
            term_overlap = len(query_terms.intersection(response_terms)) / len(query_terms)
        else:
            term_overlap = 0.0
            
        # Check response structure
        has_paragraphs = '\n\n' in response_text
        has_bullet_points = '- ' in response_text or '* ' in response_text
        has_numbered_list = bool(re.search(r'\n\d+\.', response_text))
        has_sections = bool(re.search(r'\n#+\s', response_text))
        
        # Calculate overall structure score
        structure_score = sum([
            has_paragraphs, 
            has_bullet_points, 
            has_numbered_list,
            has_sections
        ]) / 4.0
        
        # Calculate estimated quality score
        relevance_score = min(1.0, term_overlap * 1.5)  # Scale up but cap at 1.0
        length_score = min(1.0, response_length / 500)  # Consider responses up to 500 chars
        
        # Final weighted score
        quality_score = 0.5 * relevance_score + 0.3 * structure_score + 0.2 * length_score
        
        # Determine if the response passes quality threshold
        passes_threshold = quality_score >= 0.6
        
        return {
            "quality_score": quality_score,
            "relevance_score": relevance_score,
            "structure_score": structure_score,
            "length_score": length_score,
            "response_length": response_length,
            "term_overlap": term_overlap,
            "passes_threshold": passes_threshold,
            "response_type": "tuple" if isinstance(response, tuple) else "string"
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract significant words from text, filtering out common stop words."""
        # Handle case where text is a tuple (e.g., from process_query return value)
        if isinstance(text, tuple) and len(text) > 0:
            text = text[0]  # Extract the first element (string) from the tuple
        
        # Handle case where text is not a string
        if not isinstance(text, str):
            self.logger.warning(f"Expected string for keyword extraction but got {type(text)}. Converting to string.")
            try:
                text = str(text)
            except Exception as e:
                self.logger.error(f"Failed to convert {type(text)} to string: {e}")
                return []
        
        # Simple stop words list
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                     'when', 'where', 'how', 'why', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'to', 'at', 'by', 'for',
                     'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out',
                     'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'all',
                     'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                     'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                     'can', 'will', 'just', 'should', 'now', 'of', 'this', 'that'}
        
        # Extract words, convert to lowercase, and filter out stop words and short words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [word for word in words if word not in stop_words]

# Attempt to import VoxSigil Evaluator
VOXSIGIL_EVALUATOR_AVAILABLE = False

try:
    # Try to import from VoxSigilRag package
    from VoxSigilRag.voxsigil_evaluator import VoxSigilResponseEvaluator, VoxSigilConfig
    VOXSIGIL_EVALUATOR_AVAILABLE = True
    logger_evaluation.info("Successfully imported VoxSigilResponseEvaluator from VoxSigilRag package")
except ImportError:
    # Try direct module import
    try:
        # Add the parent directory to the path
        parent_dir = Path(__file__).resolve().parent.parent.parent
        if parent_dir.exists():
            sys.path.append(str(parent_dir))
            try:
                from VoxSigilRag.voxsigil_evaluator import VoxSigilEvaluator
                VOXSIGIL_EVALUATOR_AVAILABLE = True
                logger_evaluation.info("Successfully imported VoxSigilEvaluator from parent directory")
            except ImportError as e:
                logger_evaluation.error(f"Failed to import VoxSigilEvaluator: {e}")
        else:
            logger_evaluation.error(f"Parent directory not found: {parent_dir}")    
    except Exception as e:
        logger_evaluation.error(f"Failed to import VoxSigilResponseEvaluator: {e}")


class ResponseEvaluator:
    """
    Evaluates responses for quality, relevance, and correctness.
    """
    
    def __init__(self):
        """Initialize the ResponseEvaluator."""
        self.logger = logger_evaluation
        self.external_evaluator = None
        
        # Try to initialize VoxSigilEvaluator if available
        if VOXSIGIL_EVALUATOR_AVAILABLE:
            try:
                self.external_evaluator = VoxSigilEvaluator()
                self.logger.info("Successfully initialized VoxSigilEvaluator")
            except Exception as e:
                self.logger.error(f"Failed to initialize VoxSigilEvaluator: {e}")
        else:
            self.logger.warning("VoxSigilEvaluator not available, using basic evaluation only")
    
    def evaluate_response(self, 
                         query: str, 
                         response: str, 
                         context: Optional[str] = None,
                         expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluates a response for quality, relevance, and correctness.
        
        Args:
            query: The original user query.
            response: The generated response to evaluate.
            context: Optional context that was used to generate the response.
            expected_answer: Optional expected answer for comparison.
            
        Returns:
            Dictionary containing evaluation results.
        """
        # Start with basic evaluation using heuristics
        results = self._basic_evaluation(query, response, context)
        
        # If VoxSigilEvaluator is available, use it for more sophisticated evaluation
        if self.external_evaluator:
            try:
                external_results = self._external_evaluation(query, response, context, expected_answer)
                # Merge with our basic results, with external results taking precedence
                results.update(external_results)
                self.logger.info(f"Used VoxSigilEvaluator to evaluate response")
            except Exception as e:
                self.logger.error(f"Error using VoxSigilEvaluator for evaluation: {e}")
        
        # Calculate overall score
        results["overall_score"] = self._calculate_overall_score(results)
        
        # Determine success threshold
        success_threshold = 0.7  # Default threshold
        results["success"] = results["overall_score"] >= success_threshold
        
        # Generate summary
        results["summary"] = self._generate_evaluation_summary(results)
        
        return results
    
    def _basic_evaluation(self, 
                         query: str, 
                         response: str, 
                         context: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs basic response evaluation using heuristics.
        
        Args:
            query: The original user query.
            response: The generated response to evaluate.
            context: Optional context that was used to generate the response.
            
        Returns:
            Dictionary containing basic evaluation results.
        """
        results = {}
        
        # Check response length
        min_expected_length = max(30, len(query) // 3)
        max_expected_length = max(10000, len(query) * 20)
        response_length = len(response)
        
        if response_length < min_expected_length:
            results["length_assessment"] = {"score": 0.3, "reason": "Response too short"}
        elif response_length > max_expected_length:
            results["length_assessment"] = {"score": 0.6, "reason": "Response excessively long"}
        else:
            results["length_assessment"] = {"score": 1.0, "reason": "Response length appropriate"}
        
        # Check if response contains query terms (relevance indicator)
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        response_terms = set(re.findall(r'\b\w+\b', response.lower()))
        
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
        query_terms = query_terms - stop_words
        
        # Calculate term overlap
        if query_terms:
            overlap_ratio = len(query_terms.intersection(response_terms)) / len(query_terms)
            if overlap_ratio < 0.3:
                results["relevance_assessment"] = {"score": 0.4, "reason": "Low term overlap with query"}
            elif overlap_ratio < 0.6:
                results["relevance_assessment"] = {"score": 0.7, "reason": "Moderate term overlap with query"}
            else:
                results["relevance_assessment"] = {"score": 0.9, "reason": "High term overlap with query"}
        else:
            results["relevance_assessment"] = {"score": 0.5, "reason": "Unable to assess term overlap"}
        
        # Check for specificity (avoid vague responses)
        if re.search(r'\b(depends|various|several|many|different|etc\.)\b', response.lower()):
            specificity_score = 0.7  # Potential indicator of generality, but not definitive
        else:
            specificity_score = 0.9
        
        results["specificity_assessment"] = {"score": specificity_score, "reason": "Based on specificity indicators"}
        
        # Check structure and formatting
        has_sections = bool(re.search(r'\n\s*\d+\.|\n\s*|\n\s*\*|\n\s*-', response))
        has_paragraphs = response.count('\n\n') > 0
        
        if has_sections:
            results["structure_assessment"] = {"score": 1.0, "reason": "Well-structured with clear sections"}
        elif has_paragraphs:
            results["structure_assessment"] = {"score": 0.8, "reason": "Contains paragraph breaks"}
        else:
            results["structure_assessment"] = {"score": 0.5, "reason": "Limited structure or formatting"}
        
        return results
    
    def _external_evaluation(self, 
                            query: str, 
                            response: str, 
                            context: Optional[str] = None,
                            expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs evaluation using the external VoxSigilEvaluator.
        
        Args:
            query: The original user query.
            response: The generated response to evaluate.
            context: Optional context that was used to generate the response.
            expected_answer: Optional expected answer for comparison.
            
        Returns:
            Dictionary containing external evaluation results.
        """
        if not self.external_evaluator:
            return {}
        
        results = {}
        
        # Prepare inputs for VoxSigilEvaluator
        eval_inputs = {
            "query": query,
            "response": response
        }
        
        if context:
            eval_inputs["context"] = context
            
        if expected_answer:
            eval_inputs["expected_answer"] = expected_answer
        
        # Call VoxSigilEvaluator
        eval_result = self.external_evaluator.evaluate(**eval_inputs)
        
        # Map VoxSigilEvaluator results to our format
        # The actual mapping would depend on what VoxSigilEvaluator returns
        if isinstance(eval_result, dict):
            # Extract useful metrics, with some error handling
            try:
                if "quality_score" in eval_result:
                    results["quality_assessment"] = {
                        "score": float(eval_result["quality_score"]),
                        "reason": eval_result.get("quality_reason", "External evaluator assessment")
                    }
                
                if "relevance_score" in eval_result:
                    results["relevance_assessment_ext"] = {
                        "score": float(eval_result["relevance_score"]),
                        "reason": eval_result.get("relevance_reason", "External evaluator assessment")
                    }
                
                if "correctness_score" in eval_result:
                    results["correctness_assessment"] = {
                        "score": float(eval_result["correctness_score"]),
                        "reason": eval_result.get("correctness_reason", "External evaluator assessment")
                    }
                
                # Add any overall score if provided
                if "overall_score" in eval_result:
                    results["external_overall_score"] = float(eval_result["overall_score"])
                
                # Include any external evaluation summary
                if "summary" in eval_result:
                    results["external_summary"] = eval_result["summary"]
            except (ValueError, TypeError) as e:
                self.logger.error(f"Error processing external evaluation results: {e}")
        
        return results
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """
        Calculates an overall score based on individual assessment scores.
        
        Args:
            results: Dictionary containing evaluation results.
            
        Returns:
            Float representing the overall score.
        """
        # Extract scores from assessment results
        scores = []
        
        # Basic assessments
        for key in ["length_assessment", "relevance_assessment", 
                   "specificity_assessment", "structure_assessment"]:
            if key in results and "score" in results[key]:
                scores.append(results[key]["score"])
        
        # External assessments (higher weight)
        for key in ["quality_assessment", "relevance_assessment_ext", "correctness_assessment"]:
            if key in results and "score" in results[key]:
                # Give external assessments more weight
                scores.append(results[key]["score"] * 1.5)
        
        # Use external overall score if available (highest weight)
        if "external_overall_score" in results:
            scores.append(results["external_overall_score"] * 2)
        
        # Calculate weighted average
        if scores:
            weighted_score = sum(scores) / (len(scores) + sum(0.5 for key in ["quality_assessment", 
                                                                            "relevance_assessment_ext", 
                                                                            "correctness_assessment"] 
                                           if key in results) + 
                                sum(1.0 for key in ["external_overall_score"] if key in results))
            return min(1.0, weighted_score)  # Cap at 1.0
        else:
            return 0.5  # Default if no scores available
    
    def _generate_evaluation_summary(self, results: Dict[str, Any]) -> str:
        """
        Generates a human-readable summary of the evaluation results.
        
        Args:
            results: Dictionary containing evaluation results.
            
        Returns:
            String summarizing the evaluation.
        """
        if "external_summary" in results:
            return results["external_summary"]
        
        # Generate a summary based on the assessment results
        overall_score = results.get("overall_score", 0)
        success = results.get("success", False)
        
        if success:
            quality_level = "excellent" if overall_score > 0.9 else "good" if overall_score > 0.8 else "acceptable"
            summary = f"Response is {quality_level} with an overall score of {overall_score:.2f}."
        else:
            quality_level = "poor" if overall_score < 0.5 else "below acceptable standards"
            summary = f"Response is {quality_level} with an overall score of {overall_score:.2f}."
        
        # Add specific feedback
        feedback_points = []
        
        for key, label in [
            ("length_assessment", "Length"),
            ("relevance_assessment", "Relevance"),
            ("specificity_assessment", "Specificity"),
            ("structure_assessment", "Structure")
        ]:
            if key in results and "reason" in results[key]:
                feedback_points.append(f"{label}: {results[key]['reason']}")
        
        if feedback_points:
            summary += " " + " ".join(feedback_points)
        
        return summary

    # Add compatibility with SimpleEvaluator.evaluate() method
    def evaluate(self, query: str, response: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Compatibility wrapper for SimpleEvaluator.evaluate() method.
        
        Args:
            query: The original user query.
            response: The generated response to evaluate.
            context: Optional context that was used to generate the response.
            
        Returns:
            Dictionary containing evaluation results.
        """
        # For backward compatibility, use evaluate_response
        return self.evaluate_response(query, response, context)
