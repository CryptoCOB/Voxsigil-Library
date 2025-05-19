# voxsigil_supervisor/scaffold_router.py
"""
Defines the scaffold router component for the VoxSigil Supervisor.
The router selects appropriate reasoning scaffolds based on the query context.
"""
from typing import Dict, List, Any, Optional
import logging
import re
import random
from pathlib import Path
import json

# Setup the logger
logger_scaffold_router = logging.getLogger("VoxSigilSupervisor.scaffold_router")

class BasicScaffoldRouter:
    """
    A simplified implementation of the ScaffoldRouter that provides basic reasoning scaffolds.
    """
    
    def __init__(self):
        """Initialize the BasicScaffoldRouter."""
        self.logger = logger_scaffold_router
        self.default_scaffold = self._get_default_scaffold()
        self.logger.info("Initialized BasicScaffoldRouter with default scaffold template")
        
    def _get_default_scaffold(self) -> str:
        """Get the default reasoning scaffold template."""
        return """
# Problem Analysis
{query}

## Step 1: Clarify the problem
- What is being asked?
- What are the key requirements?

## Step 2: Analyze relevant information
- Key concepts: {concepts}
- Important constraints: {constraints}

## Step 3: Develop a structured solution approach
1. {step1}
2. {step2}
3. {step3}

## Step 4: Verify and evaluate
- Does the solution meet all requirements?
- Improvements or optimizations?
"""
    
    def select_scaffold(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Select an appropriate scaffold for the given query and context.
        
        Args:
            query: The user's query.
            context: Optional additional context.
            
        Returns:
            A dictionary with 'id' and 'content' keys for the selected scaffold.
        """
        self.logger.info("Using default reasoning scaffold")
        return {
            "id": "default_basic_scaffold", 
            "content": self.default_scaffold
        }
    
    def apply_scaffold(self, scaffold: str, query: str, context: Optional[str] = None) -> str:
        """
        Apply the scaffold template to the given query and context.
        
        Args:
            scaffold: The scaffold template.
            query: The user's query.
            context: Optional additional context.
            
        Returns:
            The filled scaffold.
        """
        # Extract key concepts (simple keyword extraction)
        concepts = self._extract_concepts(query)
        
        # Extract potential constraints
        constraints = self._extract_constraints(query, context)
        
        # Generate simple steps
        steps = self._generate_steps(query)
        
        # Fill the template
        filled_scaffold = scaffold.format(
            query=query,
            concepts=", ".join(concepts) if concepts else "N/A",
            constraints=", ".join(constraints) if constraints else "N/A",
            step1=steps[0] if steps else "Analyze the problem",
            step2=steps[1] if len(steps) > 1 else "Develop solution approach",
            step3=steps[2] if len(steps) > 2 else "Implement and test"
        )
        
        return filled_scaffold
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simple keyword extraction based on noun phrases
        words = re.findall(r'\b[A-Za-z][A-Za-z-]+\b', text)
        # Filter to keep only potentially meaningful concepts (longer words)
        return [w for w in words if len(w) > 5][:3]  # Limit to top 3
        
    def _extract_constraints(self, query: str, context: Optional[str] = None) -> List[str]:
        """Extract potential constraints from query and context."""
        constraint_indicators = ["must", "should", "requirement", "required", "constraint", "limit"]
        constraints = []
        
        # Check for constraint indicators in the query
        for indicator in constraint_indicators:
            if indicator in query.lower():
                # Extract the sentence containing the indicator
                sentences = re.split(r'[.!?]', query)
                for sentence in sentences:
                    if indicator in sentence.lower():
                        constraints.append(sentence.strip())
                        break
        
        return constraints[:2]  # Limit to top 2 constraints
        
    def _generate_steps(self, query: str) -> List[str]:
        """Generate simple step descriptions based on the query."""
        # Prepare basic steps based on query length and complexity
        if len(query) < 50:  # Short query
            return ["Understand the key requirements", 
                    "Develop a concise solution", 
                    "Validate the approach"]
        else:  # Longer, potentially more complex query
            return ["Break down the problem into components", 
                    "Analyze each component systematically", 
                    "Synthesize findings into a comprehensive solution"]

class ScaffoldRouter:
    """
    The ScaffoldRouter selects appropriate reasoning scaffolds based on 
    query content, context, and evaluation feedback.
    """
    
    def __init__(self, scaffolds_dir: Optional[Path] = None):
        """
        Initialize the ScaffoldRouter.
        
        Args:
            scaffolds_dir: Optional directory containing scaffold templates.
                           If None, will use default scaffolds from the package.
        """
        self.logger = logger_scaffold_router
        
        if scaffolds_dir is None:
            # Default to a scaffolds directory in the package
            self.scaffolds_dir = Path(__file__).resolve().parent / "resources" / "scaffolds"
        else:
            self.scaffolds_dir = Path(scaffolds_dir)
        
        # Initialize scaffold templates
        self.scaffolds = self._load_scaffolds()
        self.scaffold_categories = self._categorize_scaffolds()
        
        self.logger.info(f"Initialized ScaffoldRouter with {len(self.scaffolds)} scaffolds")
    
    def _load_scaffolds(self) -> Dict[str, str]:
        """
        Load scaffold templates from files or define defaults if not available.
        
        Returns:
            Dictionary mapping scaffold IDs to scaffold templates.
        """
        scaffolds = {}
        
        # First try to load from files
        if self.scaffolds_dir.exists():
            for file_path in self.scaffolds_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        scaffold_data = json.load(f)
                        for scaffold_id, scaffold_content in scaffold_data.items():
                            scaffolds[scaffold_id] = scaffold_content
                except Exception as e:
                    self.logger.error(f"Failed to load scaffold file {file_path}: {e}")
        
        # If no scaffolds were loaded, use these defaults
        if not scaffolds:
            self.logger.warning("No scaffold files found, using default scaffolds")
            scaffolds = {
                "step_by_step": "Approach this problem step-by-step:\n"
                                "1. Carefully analyze what the query is asking for\n"
                                "2. Extract key information from the provided context\n"
                                "3. Determine the necessary steps to solve the problem\n"
                                "4. Work through each step methodically\n"
                                "5. Verify your solution against the requirements",
                
                "poe_scaffold": "To solve this problem effectively:\n"
                                "- Problem understanding: Restate the problem in my own words\n"
                                "- Observation: Note the key information provided\n"
                                "- Extraction: Identify the relevant context needed\n"
                                "- Solution approach: Outline how I will solve this",
                
                "chain_of_thought": "I'll use chain-of-thought reasoning:\n"
                                    "- First, I'll understand what is being asked\n"
                                    "- Then, I'll analyze the information provided\n"
                                    "- Next, I'll reason through intermediate steps\n"
                                    "- Finally, I'll arrive at the answer",
                
                "structured_analysis": "Structured Analysis Approach:\n"
                                       "1. CLARIFY: What exactly is the question asking?\n"
                                       "2. IDENTIFY: What are the relevant pieces of information?\n"
                                       "3. ANALYZE: How do these pieces connect to form a solution?\n"
                                       "4. VERIFY: Does my solution address the original question?",
                
                "hypothesis_testing": "I'll use a hypothesis testing approach:\n"
                                      "1. Form an initial hypothesis based on the query\n"
                                      "2. Identify what evidence would confirm or refute this hypothesis\n"
                                      "3. Check the context for this evidence\n"
                                      "4. Refine my hypothesis accordingly\n"
                                      "5. Conclude with the best-supported answer"
            }
        
        return scaffolds
    
    def _categorize_scaffolds(self) -> Dict[str, List[str]]:
        """
        Categorize scaffolds by their purpose/type for better selection.
        
        Returns:
            Dictionary mapping categories to lists of scaffold IDs.
        """
        categories = {
            "general": ["step_by_step", "chain_of_thought"],
            "analytical": ["structured_analysis", "hypothesis_testing"],
            "extraction": ["poe_scaffold"],
            "complex": ["structured_analysis", "hypothesis_testing"],
            "simple": ["step_by_step", "chain_of_thought"],
            "fallback": ["step_by_step"]
        }
        
        # Ensure all categories contain only valid scaffold IDs
        for category, scaffold_ids in list(categories.items()):
            valid_ids = [sid for sid in scaffold_ids if sid in self.scaffolds]
            if len(valid_ids) != len(scaffold_ids):
                self.logger.warning(f"Some scaffold IDs in category '{category}' are invalid")
            categories[category] = valid_ids
            
            # If a category is empty, remove it
            if not valid_ids:
                del categories[category]
        
        return categories
    
    def select_scaffold(self, 
                       query: str, 
                       context: Optional[str] = None,
                       task_type: Optional[str] = None,
                       prev_scaffold_id: Optional[str] = None,
                       feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Select an appropriate scaffold based on the query and context.
        
        Args:
            query: The user query.
            context: Optional context information.
            task_type: Optional task type hint.
            prev_scaffold_id: Optional ID of previously used scaffold.
            feedback: Optional feedback from previous attempts.
            
        Returns:
            Dictionary with 'id' and 'content' keys for the selected scaffold.
        """
        # Determine query characteristics
        query_length = len(query.split())
        is_complex = query_length > 50 or (context and len(context.split()) > 200)
        
        # Determine appropriate category based on query and context
        category = "general"  # Default category
        
        # Check for analytical questions
        analytical_patterns = [
            r'\banalyze\b', r'\bcompare\b', r'\bcontrast\b', r'\bevaluate\b',
            r'\bexplain\b', r'\bwhy\b', r'\bhow\b', r'\bwhat\s+if\b'
        ]
        if any(re.search(pattern, query.lower()) for pattern in analytical_patterns):
            category = "analytical"
        
        # Check for extraction/information retrieval
        extraction_patterns = [
            r'\bfind\b', r'\bextract\b', r'\bget\b', r'\bidentify\b',
            r'\bwhat\s+is\b', r'\bwho\s+is\b', r'\bwhen\s+did\b', r'\bwhere\s+is\b'
        ]
        if any(re.search(pattern, query.lower()) for pattern in extraction_patterns):
            category = "extraction"
        
        # Override based on task_type if provided
        if task_type:
            if task_type.lower() in ["analysis", "reasoning", "complex"]:
                category = "analytical"
            elif task_type.lower() in ["extraction", "retrieval", "lookup"]:
                category = "extraction"
        
        # Override based on complexity
        if is_complex:
            category = "complex"
        elif query_length < 20 and not is_complex:
            category = "simple"
        
        # Consider feedback if available
        if feedback:
            success = feedback.get("success", False)
            if not success and prev_scaffold_id:
                # If previous attempt failed, try a different scaffold
                self.logger.info(f"Previous scaffold {prev_scaffold_id} unsuccessful, trying different category")
                # Try to avoid the previous category
                current_category = next((c for c, ids in self.scaffold_categories.items() 
                                        if prev_scaffold_id in ids), None)
                if current_category and current_category != "fallback":
                    # Try a different category
                    available_categories = [c for c in self.scaffold_categories.keys() 
                                          if c != current_category and c != "fallback"]
                    if available_categories:
                        category = random.choice(available_categories)
                        self.logger.info(f"Switched to category: {category}")
        
        # Select a scaffold from the appropriate category
        scaffold_ids = self.scaffold_categories.get(category, self.scaffold_categories["fallback"])
        
        # Avoid using the same scaffold if possible
        if prev_scaffold_id and prev_scaffold_id in scaffold_ids and len(scaffold_ids) > 1:
            scaffold_ids = [sid for sid in scaffold_ids if sid != prev_scaffold_id]
        
        # Pick a scaffold
        scaffold_id = random.choice(scaffold_ids)
        scaffold_content = self.scaffolds[scaffold_id]
        
        self.logger.info(f"Selected scaffold {scaffold_id} from category {category}")
        
        return {
            "id": scaffold_id,
            "content": scaffold_content,
            "category": category
        }
    
    def customize_scaffold(self, 
                          scaffold: Dict[str, Any], 
                          query: str, 
                          context: Optional[str] = None) -> Dict[str, Any]:
        """
        Customize a scaffold template based on the specific query and context.
        
        Args:
            scaffold: The scaffold dictionary from select_scaffold.
            query: The user query.
            context: Optional context information.
            
        Returns:
            Updated scaffold dictionary with customized content.
        """
        # This is a simple implementation - in a real system, this could use
        # an LLM to tailor the scaffold more specifically to the query
        
        # For now, we'll just make some simple substitutions
        content = scaffold["content"]
        
        # Add query-specific elements if detected
        query_lower = query.lower()
        
        # For numerical questions, add calculation step
        if any(term in query_lower for term in ["calculate", "compute", "how many", "sum", "average"]):
            if "step-by-step" in content:
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if "steps" in line.lower() or "methodically" in line.lower():
                        lines.insert(i+1, "   - Show your calculations clearly")
                        content = "\n".join(lines)
                        break
        
        # For comparison questions, emphasize comparison points
        if any(term in query_lower for term in ["compare", "contrast", "difference", "similar"]):
            if not "comparison" in content.lower():
                content += "\n- Make sure to explicitly state key points of comparison"
        
        # Return the updated scaffold
        scaffold["content"] = content
        scaffold["customized"] = True
        
        return scaffold
