# voxsigil_supervisor/utils/sigil_formatting.py
"""
Utilities for formatting VoxSigil constructs (sigils, scaffolds) for prompts or display.
This logic is adapted from your existing VoxSigilRAG.format_sigil_for_prompt.
"""
from typing import List, Dict, Any
from pathlib import Path 

# This function is essentially a copy/adaptation of your
# VoxSigilRAG.format_sigil_for_prompt method, made standalone.
def format_sigil_detail(sigil_data: Dict[str, Any], detail_level: str = "standard") -> str:
    """
    Formats a single VoxSigil construct's data into a string.
    """
    output = []
    
    # Identifier: Prefer 'sigil' (glyph), then 'name', then 'id'
    identifier = sigil_data.get('sigil')
    id_type = "Sigil"
    if not identifier:
        identifier = sigil_data.get('name', sigil_data.get('id', 'UnknownConstruct'))
        id_type = "Construct" # If no 'sigil' glyph field

    output.append(f"{id_type}: \"{identifier}\"")

    # Consolidate tag handling from your VoxSigilRAG.format_sigil_for_prompt
    all_tags = []
    if 'tag' in sigil_data and sigil_data['tag']:
        if isinstance(sigil_data['tag'], str) and sigil_data['tag'] not in all_tags:
            all_tags.append(sigil_data['tag'])
    if 'tags' in sigil_data and sigil_data['tags']:
        tags_val = sigil_data['tags']
        if isinstance(tags_val, list):
            for t_item in tags_val:
                if isinstance(t_item, str) and t_item not in all_tags:
                    all_tags.append(t_item)
        elif isinstance(tags_val, str) and tags_val not in all_tags:
            all_tags.append(tags_val)
    if all_tags:
        output.append(f"Tags: {', '.join(f'\"{tag}\"' for tag in all_tags)}")

    if 'principle' in sigil_data and isinstance(sigil_data['principle'], str):
        output.append(f"Principle: \"{sigil_data['principle']}\"")
    
    if detail_level.lower() == "summary":
        return '\n'.join(output)
    
    # Standard Detail Level
    if 'usage' in sigil_data and isinstance(sigil_data['usage'], dict):
        usage_info = sigil_data['usage']
        if 'description' in usage_info and isinstance(usage_info['description'], str):
            output.append(f"Usage: \"{usage_info['description']}\"")
        if 'examples' in usage_info and usage_info['examples']:
            examples_val = usage_info['examples']
            first_example_str = ""
            if isinstance(examples_val, list) and examples_val:
                first_example_str = str(examples_val[0])
            elif isinstance(examples_val, str):
                first_example_str = examples_val
            if first_example_str:
                 output.append(f"Example: \"{first_example_str[:250]}{'...' if len(first_example_str) > 250 else ''}\"") # Truncate long examples
    
    if '_source_file' in sigil_data and isinstance(sigil_data['_source_file'], str):
        try:
            output.append(f"Source File: {Path(sigil_data['_source_file']).name}")
        except Exception: # Handle if _source_file is not a valid path string
             output.append(f"Source File Ref: {sigil_data['_source_file']}")


    # Full Detail Level
    if detail_level.lower() == "full":
        if 'relationships' in sigil_data and isinstance(sigil_data['relationships'], dict):
            for rel_type, rel_values in sigil_data['relationships'].items():
                if rel_values: # Ensure there are values to format
                    values_str_list = []
                    if isinstance(rel_values, list):
                        values_str_list = [f'"{rv}"' for rv in rel_values if isinstance(rv, str)]
                    elif isinstance(rel_values, str):
                        values_str_list = [f'"{rel_values}"']
                    
                    if values_str_list:
                        output.append(f"Relationship ({rel_type}): {', '.join(values_str_list)}")
            
        if 'prompt_template' in sigil_data and isinstance(sigil_data['prompt_template'], dict):
            pt_info = sigil_data['prompt_template']
            if 'type' in pt_info: output.append(f"Template Type: {pt_info['type']}")
            if 'description' in pt_info: output.append(f"Template Description: \"{pt_info['description']}\"")
            # Generally avoid including full template content in RAG context unless specifically needed and small

    return '\n'.join(output)


def format_sigils_for_prompt(sigil_list: List[Dict[str, Any]], detail_level: str = "standard") -> str:
    """
    Formats a list of Voxsigil construct dictionaries into a single string for prompt injection.
    """
    if not sigil_list:
        return "No relevant Voxsigil constructs found for context."

    formatted_parts = [format_sigil_detail(s_data, detail_level) for s_data in sigil_list if isinstance(s_data, dict)]
    return "\n\n---\n\n".join(filter(None, formatted_parts)) # Use a clear separator and filter empty strings