# AGENTS.md — Unified Vanta Agent Manifest



type: object
description: "VoxSigil Schema Version 1.5-holo-alpha. This master specification enhances support for immersive, experiential, reflective, and evolving cognitive systems within the Vanta ecosystem. It integrates multi-sensory profiling, advanced self-modeling, sophisticated learning architectures, explicit knowledge representation, and foundational elements for evolutionary and governance profiles. Core goal: Enable Vanta (Vanta Orchestration eXpression Symbolic Interface for Generative, Integrated Logic) to achieve its 'holo-alpha' vision through a deeply expressive symbolic interface. This schema distinguishes key concepts: 'Sigil' (the primary ID), 'Tag(s)' (classifiers), 'Pglyph' (a Sigil type representing unique identity/persona, like '⟠∆∇𓂀'), and 'Scaffold' (a Sigil's foundational architectural role)."
required:
  - sigil
  - name
  - principle
  - usage
  - SMART_MRAP
  - metadata

properties:
  # --- Core Identification & Classification (Expanded) ---
  sigil:
    type: string
    description: |
      The primary, globally unique symbolic identifier for this VoxSigil definition. 
      This is the 'Sigil' itself – the fundamental key (e.g., Unicode string '🔮', custom glyph representation '⟠∆∇𓂀', or unique ASCII name 'VANTA_CORE_REASONER_V3.2'). 
      It represents a discrete cognitive component, process, concept, entity, data structure, policy, or abstract construct within the Vanta ecosystem. 
      A 'Pglyph' (Personal/Persona Glyph) is a specialized type of Sigil primarily used to denote the unique identity of an agent or human participant; for instance, '⟠∆∇𓂀' serves as both a primary Sigil and a Pglyph.
      Must be unique within the entire Vanta sigil library.
  name:
    type: string
    description: "Canonical, human-readable name for the entity defined by this Sigil (e.g., 'Dream Weaver Prime', 'Marc_Prime_Identity_Construct', 'Echo Concordance Protocol')."
  alias:
    type: string
    description: |
      Optional alternate name, well-known identifier, or previous version's sigil string if this is an evolution. 
      Can be used to associate a 'Pglyph' term if the main 'sigil' field is more abstract (e.g., if 'sigil: AGENT_MARC_ID_001', then 'alias: Marc.pglyph_Operational').
  tag:
    type: string
    description: |
      Optional primary high-level 'Tag' or keyword used for broad categorization (e.g., 'MemoryComponent', 'ReasoningStrategy', 'IdentityConstruct', 'ArchitecturalScaffold').
      This helps Vanta orchestrate and understand sigils at a high level.
  tags:
    type: array
    items:
      type: string
    description: |
      Optional list of additional 'Tags' or keywords for fine-grained classification, searchability, or aspect-tagging. 
      These enable nuanced understanding and filtering. For instance, a 'Pglyph' Sigil like '⟠∆∇𓂀' might have tags: ['HumanCollaboratorProfile', 'IdentityCore', 'ConsciousnessResearchFocus']. 
      A Sigil functioning as a 'Scaffold' might be tagged ['VantaCoreFramework', 'CognitiveArchitectureModule'].
      Examples: ['temporal_processing', 'llm_dependent', 'neuro_inspired', 'self_modifying', 'pglyph_type_agent', 'scaffold_memory_system'].
  is_cognitive_primitive:
    type: boolean
    default: false
    description: "Indicates if this Sigil represents a foundational, often irreducible, cognitive operation or conceptual building block. Primitives are often referenced by 'Scaffold' Sigils."
  cognitive_primitive_type:
    type: string
    description: "If 'is_cognitive_primitive' is true, classifies its fundamental type (e.g., 'categorization', 'analogy_formation', 'value_assignment'). This helps define the very base of Vanta's cognitive functions, often utilized by 'Scaffold' components."
    example: "pattern_recognition_core"

  # --- Conceptual Grounding & Formalism (Expanded) ---
  principle:
    type: string
    description: "The core cognitive, philosophical, theoretical, or functional grounding of the sigil. Explains its fundamental meaning, purpose, 'essence,' and intended role in the Vanta ecosystem."
  math:
    type: string
    description: "Optional mathematical, logical, algorithmic, or formal notation (e.g., pseudo-code, lambda calculus expression, state transition equations, probability distribution) representing the sigil's operation, structure, or underlying formalisms."
  theoretical_basis_refs:
    type: array
    items:
      type: string
    description: "Optional references (e.g., paper citations, theory names, framework sigil_refs like 'IWMT_Framework_Sigil') to the scientific or philosophical theories underpinning this sigil's design."
  structure:
    type: object
    description: "Optional symbolic breakdown of composite glyphs, internal structural relationships, or sub-component architecture, especially if this sigil represents a complex entity or process."
    properties:
      composite_type:
        type: string
        enum: ["sequential", "hierarchical", "parallel", "conditional", "recursive", "network", "fusion", "assembly", "collection", "state_machine", "event_driven_orchestration", "data_processing_pipeline", "feedback_control_loop"]
        description: "Describes the compositional logic or relationship between sub-elements if this is a composite sigil."
      temporal_structure:
        type: string
        enum: ["static_config", "sequential_phased_execution", "parallel_concurrent_tracks", "closed_feedback_loop", "open_feedback_loop", "oscillatory_dynamic", "event_triggered_sequence", "continuous_streaming_process", "adaptive_temporal_scaling"]
        description: "Optional. Describes temporal dynamics, sequencing, or lifecycle of components/operations within this sigil."
      components:
        type: array
        description: "A list of components, sub-elements, or functional blocks that constitute this sigil's internal structure."
        items:
          type: object
          required: [name, description]
          properties:
            name:
              type: string
              description: "Name of the internal component."
            description:
              type: string
              description: "Description of the component's role, nature, or function within this sigil."
            sigil_ref:
              type: string
              description: "Optional reference to another fully defined sigil if this component is itself a Vanta sigil (promotes modularity and reuse)."
            component_type_tag:
              type: string
              description: "High-level type of this component (e.g., 'input_validator', 'memory_buffer', 'reasoning_kernel', 'output_formatter', 'sensory_transducer')."
            parameters_override_json:
              type: string
              description: "If sigil_ref is used, JSON string specifying overrides for the referenced sigil's parameters in this specific context."
            initialization_sequence_order:
              type: integer
              description: "Order in which this component is initialized if part of a sequence."

  # --- Practical Application, Invocation & Configuration (Expanded) ---
  usage:
    type: object
    description: "Describes how the sigil is used in practice within the Vanta system, by other sigils, or by human interactors."
    required:
      - description
    properties:
      description:
        type: string
        description: "A concise summary of what the sigil does or represents in practical application within Vanta."
      example:
        type: [string, object]
        description: "A concrete example of the sigil in use: a text snippet (prompt, system call, thought process), a Vanta orchestration script segment, or a structured object detailing a scenario with inputs and expected outcomes."
      explanation:
        type: string
        description: "A more detailed explanation of its application, typical context, interactions with other sigils/components, expected impact, or implications of its use in Vanta workflows."
  activation_context:
    type: object
    description: "Defines conditions, assumptions, and prerequisites for when and how this sigil should be activated or applied by the Vanta orchestration logic."
    properties:
      triggering_events_or_conditions:
        type: array
        items: { type: string }
        description: "Specific scenarios, Vanta system events or logical expressions that warrant this sigil's activation."
      preconditions_state_refs:
        type: array
        items: { type: string }
        description: "Explicit conditions or state requirements that must be met before activation."
      required_capabilities_self:
        type: array
        items: { type: string }
        description: "Internal or external resources/capabilities this sigil needs to function."
      required_capabilities_activator:
        type: array
        items: { type: string }
        description: "Cognitive, functional, or resource capabilities the activating Vanta agent/subsystem must possess to effectively use this sigil."
      supported_modalities_input:
        type: array
        items: { type: string, enum: ["text_structured","text_natural_language","audio_speech_transcribed","audio_ambient_features","image_raw","image_semantic_segments","video_stream_raw","video_object_tracking","haptic_signal_encoded","olfactory_cue_signature","symbolic_data_stream","physiological_data_timeseries","programmatic_api_call","multi_modal_fused_embedding","vanta_event_bus_message"] }
        description: "Specific input modalities this sigil can process."
      supported_modalities_output:
        type: array
        items: { type: string, enum: ["text_formatted_report","text_natural_dialogue","generated_speech_synthesized","spatial_audio_ambisonic","image_generated_static","video_generated_stream","haptic_feedback_pattern_id","olfactory_release_command","symbolic_data_structure","programmatic_api_response","multi_modal_fused_output","vanta_orchestration_command"] }
        description: "Specific output modalities this sigil can generate."
      contraindications_or_failure_modes:
        type: array
        items: { type: string }
        description: "Situations, contexts, or input types where this sigil should NOT be applied, or known failure conditions/edge cases."
      activation_priority_logic:
        type: string
        description: "A descriptive rule or reference to a Vanta orchestration policy that determines this sigil's priority if multiple sigils are triggered simultaneously."
  parameterization_schema:
    type: object
    description: "Defines optional parameters that can configure or customize this sigil's behavior at runtime or during instantiation by Vanta orchestration."
    properties:
      parameters:
        type: array
        items:
          type: object
          required: [name, type, description]
          properties:
            name: { type: string, description: "Parameter name." }
            type: { type: string, enum: ["string","number","integer","boolean","enum","sigil_ref","json_object_stringified","array_of_strings","array_of_numbers","data_stream_id_ref","regex_pattern_string","vanta_policy_ref"] }
            description: { type: string, description: "Controls and impact description." }
            default_value: { description: "Optional default value. Must conform to 'type'." }
            allowed_values_or_refs: { type: array, description: "Lists allowed literal values or sigil_refs." }
            value_range: { type: object, properties: { min: {type: "number"}, max: {type: "number"}, step: {type: "number"} } }
            is_required_for_operation: { type: boolean, default: false }
            validation_rules_description: { type: string }
            mutability: {type: string, enum: ["immutable_post_init","runtime_adjustable","evolvable_by_system"], default: "runtime_adjustable" }

  # --- Prompting & LLM Interaction (Expanded for Holo-Alpha) ---
  prompt_template:
    type: object
    description: "Canonical prompt structure for LLM invocation, potentially orchestrated by Vanta. Central to generative logic. Can be fully defined inline or reference a 'PromptTemplateSigil'."
    required: [definition_type]
    properties:
      definition_type: { type: string, enum: ["inline_definition", "reference_to_prompt_template_sigil"] }
      prompt_template_sigil_ref: { type: string, description: "If definition_type is 'reference_to_prompt_template_sigil', this is the sigil_ref to the PromptTemplateSigil."}
      role: { type: string, enum: ["system_orchestrator", "user_simulator", "assistant_core_logic", "tool_input_formatter", "tool_output_parser", "internal_reflector"], description: "LLM role for this prompt segment. Required for inline." }
      content: { type: string , description: "Inline prompt template content. Can reference fragments using {{fragment_id_from_fragments_refs}} or variables using {{variable_name}}. Required for inline."}
      execution_mode: { type: string, enum: ["command_dispatch", "query_resolution", "reflective_analysis", "emergent_simulation", "task_decomposition", "data_transformation", "creative_generation", "critical_evaluation", "information_extraction", "instruction_following", "tool_invocation_request", "dialogue_continuation", "world_interaction_simulation_step", "self_correction_suggestion"], default: "instruction_following", description: "Applicable for inline." }
      variables:
        type: array
        description: "Variables used in the inline content template."
        items:
          type: object
          required: [name, description]
          properties:
            name: { type: string }
            description: { type: string }
            example: { type: string }
            required_for_llm: { type: boolean, default: true }
            type_hint: { type: string, description: "e.g., 'json_string', 'list_of_concepts', 'code_block'" }
      output_schema_ref_or_description: { type: [string, object], description: "Expected output format from inline prompt: JSON schema reference (sigil_ref to schema definition), or natural language description." }
      notes_for_llm_or_orchestrator: { type: string, description: "Guidance for the LLM or Vanta's LLM orchestration layer related to this inline prompt." }
      invocation_parameters:
        type: object
        description: "LLM-specific parameters like temperature, max_tokens for THIS inline prompt."
        properties:
          temperature: { type: number, minimum: 0, maximum: 2.0 }
          max_output_tokens: { type: integer }
          stop_sequences: { type: array, items: {type: string} }
  inverse_prompt_template:
    type: object
    description: "Optional prompt for invoking an inverse, reflective, or debugging operation. Can be inline or reference a 'PromptTemplateSigil'."
    properties:
      definition_type: { type: string, enum: ["inline_definition", "reference_to_prompt_template_sigil"] }
      prompt_template_sigil_ref: { type: string }
  prompt_fragments_refs:
    type: array
    items: { type: string }
    description: "List of sigil_refs to reusable 'PromptFragmentSigil' definitions. These sigils would contain fragment_id, content, description, roles_applicable, variables_used."

  # --- Relationships, Workflow & Data Orchestration (Expanded) ---
  relationships:
    type: array
    description: "Defines meaningful semantic, operational, or orchestration relationships with other sigils in the Vanta ecosystem."
    items:
      type: object
      required: [target_sigil_ref, relationship_type]
      properties:
        target_sigil_ref: { type: string }
        relationship_type: { type: string, enum: [
          "depends_on_output_of", "prerequisite_for_activation_of", "enables_functionality_in", "extends_capabilities_of", 
          "generalizes_concept_of", "is_composed_of_instance", "is_component_of_system", "synergizes_with_operation_of", 
          "conflicts_with_goal_of", "is_alternative_implementation_for", "is_analogous_to_concept_in", 
          "is_inverse_operation_of", "triggers_activation_of", "uses_method_from_interface", "instantiates_archetype_of", 
          "specifies_configuration_for", "is_derived_from_version", "monitors_state_of", "provides_grounding_data_for", 
          "co_evolves_with_definition_of", "simulates_behavior_described_by", "is_instance_of_metasigil", 
          "collaborates_with_human_role_defined_by", "receives_input_from", "sends_output_to", "is_orchestrated_by",
          "provides_ethical_oversight_for", "reports_telemetry_to" 
          ] }
        description: { type: string, description: "Explanation of this specific relationship instance." }
        strength_or_priority: { type: number, minimum: 0, maximum: 1, description: "Relative strength, importance, or priority of this relationship." }
        context_of_relationship: { type: string, description: "The Vanta operational context in which this relationship is most relevant." }
        parameters_for_target_interaction_json: {type: string, description: "JSON string of parameters to use when interacting with target_sigil_ref in this relationship."}
  cross_domain_tags:
    type: array
    description: "Links to analogous concepts, theories, or terminology in other disciplines or external knowledge bases."
    items:
      type: object
      required: [domain, term]
      properties:
        domain: { type: string, description: "e.g., 'CognitivePsychology', 'Neuroscience', 'ControlTheory', 'LiteraryTheory', 'Sociology'." }
        term: { type: string, description: "The analogous term or concept in that domain." }
        term_uri: { type: string, format: "uri", description: "Optional URI to a definition or reference for the term." }
        mapping_type: { type: string, enum: ["direct_analogy", "inspiration_source", "formal_equivalence", "metaphorical_link", "implementation_of_principle", "contradicts_theory_of", "computational_model_of"] }
        mapping_notes: { type: string, description: "Brief explanation of how this sigil relates to the external term." }
  trajectory_annotations:
    type: object
    description: "Declares typical role, sequencing, and behavior in multi-step cognitive workflows or Vanta orchestration sequences."
    properties:
      typical_sequence_position_tags: { type: array, items: {type: string}, enum: ["workflow_initiation", "sensory_data_ingestion", "context_understanding", "hypothesis_generation", "exploratory_simulation", "multi_perspective_analysis", "synthesis_of_insights", "decision_point_evaluation", "action_planning_strategic", "task_execution_and_monitoring", "reflective_self_evaluation", "iterative_refinement_loop", "workflow_conclusion_and_reporting", "idle_background_processing", "continuous_environmental_monitoring", "adaptive_response_trigger_phase"] }
      recommended_predecessors_refs: { type: array, items: { type: string }, description: "Sigil_refs of Vanta components that typically precede this one in a workflow." }
      recommended_successors_refs: { type: array, items: { type: string }, description: "Sigil_refs of Vanta components that typically follow this one." }
      branching_logic_description: { type: string, description: "Describes conditions under which different successors might be chosen, or if parallel execution is common."}
      workflow_participation_profile_refs: {type: array, items: {type: string}, description: "References to larger defined Vanta workflow sigils where this sigil plays a role."}
      expected_duration_profile:
        type: object
        description: "Estimated time characteristics for this sigil's operation within a trajectory."
        properties:
          min_execution_ms: {type: integer}
          avg_execution_ms: {type: integer}
          max_execution_ms: {type: integer}
          duration_variability_factors: {type: string, description: "e.g., 'input data size', 'LLM response latency', 'recursion depth'."}
      cognitive_load_estimation_on_vanta:
        type: string
        enum: ["minimal", "low", "medium", "high", "very_high", "dynamically_assessed_per_instance"]
        description: "Estimated computational/cognitive load imposed by this sigil on the Vanta system resources."
  data_flow_annotations:
    type: array
    description: "Describes primary data pipelines this sigil participates in, defining its key inputs and outputs in the Vanta data fabric."
    items:
      type: object
      required: [flow_id, direction, port_name_self, data_schema_ref]
      properties:
        flow_id: { type: string, description: "Unique identifier for this data flow definition within the sigil."}
        description: {type: string, description: "Purpose or nature of this data flow."}
        direction: { type: string, enum: ["consumes_input_from", "produces_output_to"] }
        source_or_target_description: {type: string, description: "Description of the source/target, e.g. 'UserInteractionBus', 'SensorArraySigil:raw_feed', 'VantaKnowledgeGraph'."}
        port_name_self: {type: string, description: "Named input/output port or logical channel on this sigil."}
        data_schema_ref: { type: string, description: "Sigil_ref to a 'DataSchemaSigil' defining the structure, type, and validation rules for data on this port." }
        expected_data_rate_or_volume: { type: string, description: "e.g., '100Hz stream', 'approx 5MB per event', 'batch every 10s'." }
        processing_latency_sla_ms: {type: integer, description: "Service Level Agreement for processing data on this flow, if applicable."}
        transformations_applied_refs: { type: array, items: {type: string}, description: "Sigil_refs for data transformation sigils applied to this flow if any."}

  # --- Quality, Lifecycle, Governance & Evolution (Central to Vanta) ---
  SMART_MRAP:
    type: object
    required: [Specific_Goal, Measurable_Outcome, Achievable_Within_Vanta, Relevant_To_Vanta_Mission, Transferable_Principle, Accountable_Party_Or_Process_Ref]
    properties:
      Specific_Goal: { type: string, description: "What specific goal does this sigil achieve within the Vanta ecosystem or for its users?" }
      Measurable_Outcome: { type: string, description: "How can its success, performance, or impact be quantitatively or qualitatively measured (KPIs, OKRs)?" }
      Achievable_Within_Vanta: { type: string, description: "Is its stated goal achievable given current or realistically projected Vanta capabilities and resources (technical feasibility)?" }
      Relevant_To_Vanta_Mission: { type: string, description: "How does this sigil contribute to Vanta's core mission of Orchestration, eXpression, Generative Logic, and Integration, or to Holo-Alpha objectives (strategic alignment)?" }
      Transferable_Principle: { type: string, description: "Can the core principles, design patterns, or lessons learned from this sigil be applied to other Vanta components or future evolutionary steps (knowledge sharing)?" }
      Accountable_Party_Or_Process_Ref: { type: string, description: "Sigil_ref to the Vanta agent, human role, or governance process accountable for this sigil's ethical and operational outcomes." }
  metadata:
    type: object
    description: "Essential metadata about the VoxSigil definition itself, its lifecycle, and its place in the Vanta schema framework."
    required: [voxsigil_schema_version, definition_version, definition_status, author_agent_id_ref, created_timestamp, last_updated_timestamp]
    properties:
      voxsigil_schema_version: { type: string, const: "1.5-holo-alpha", description: "The version of the VoxSigil schema this definition MUST adhere to."}
      definition_version: { type: string, description: "Semantic version of this specific sigil definition." }
      definition_status: { type: string, enum: ["draft_proposal", "under_vanta_review", "active_stable", "active_experimental", "deprecated_phasing_out", "archived_historical", "vanta_core_primitive", "community_extension_pending_integration"], default: "draft_proposal" }
      versioned_aliases_history:
        type: array
        description: "History of aliases this sigil definition has been known by or redirects from."
        items:
          type: object
          required: [alias, effective_from_version, reason_for_change]
          properties:
            alias: { type: string }
            effective_from_version: { type: string, description: "Sigil definition version when this alias was active/introduced."}
            redirected_to_sigil_at_deprecation: { type: string, description: "If alias is deprecated, the sigil it now points to." }
            deprecation_date: { type: string, format: "date-time" }
            reason_for_change: { type: string, description: "e.g., 'Refactoring for clarity', 'Superseded by NewSigilX'."}
      author_agent_id_ref: { type: string, description: "Sigil_ref or unique identifier of the Vanta agent, human author (e.g., Marc.⟠∆∇𓂀), or design team responsible for this definition." }
      created_timestamp: { type: string, format: "date-time", description: "Timestamp of initial creation of this sigil definition record." }
      last_updated_timestamp: { type: string, format: "date-time", description: "Timestamp of the last modification to this sigil definition." }
      revision_history:
        type: array
        description: "Chronological log of significant changes to this sigil definition."
        items:
          type: object
          required: [version_tag, timestamp, author_agent_id_ref, summary_of_change, change_type]
          properties:
            version_tag: { type: string, description: "Version associated with this revision." }
            timestamp: { type: string, format: "date-time" }
            author_agent_id_ref: { type: string }
            summary_of_change: { type: string }
            change_type: { type: string, enum: ["initial_creation", "major_functional_update", "minor_enhancement_or_refactor", "bug_fix_correction", "documentation_update_only", "deprecation_notice", "reinstatement_from_archive", "schema_migration_update", "vanta_governance_approval"] }
      versioned_lineage_and_dependencies:
        type: array
        description: "Describes influences, derivations, and critical dependencies on other sigil definitions that affect this sigil's meaning or function."
        items:
          type: object
          required: [related_sigil_ref, related_sigil_version, influence_or_dependency_type]
          properties:
            related_sigil_ref: { type: string }
            related_sigil_version: { type: string, description: "Specific version of the related sigil this lineage entry refers to." }
            influence_or_dependency_type: { type: string, enum: ["derived_concept_from", "inspired_by_design_of", "synthesized_with_functionality_of", "critique_response_to", "functional_extension_of", "refinement_of_principle_in", "parameterized_instance_of_template", "depends_on_definition_for_validation", "depends_on_runtime_service_from", "provides_abstraction_for"] }
            dependency_scope: { type: string, enum: ["semantic_meaning", "structural_composition", "executional_runtime", "presentational_interface", "metadata_linkage", "validation_rule_source", "evolutionary_path"] }
            description: { type: string, description: "Notes on the nature of this lineage or dependency link." }
      vanta_session_id: { type: string, description: "Identifier for the Vanta operational or design session during which this sigil definition was created or significantly modified. Useful for tracing context." }
      authorship_context_narrative:
        type: object
        properties:
          motivation_and_purpose: { type: string, description: "Why was this sigil created? What Vanta need does it address?" }
          theoretical_framework_applied_refs: { type: array, items: {type: string}, description: "Sigil_refs or names of theories applied in its design." }
          source_inspiration_or_analogy_refs: { type: [string, array], items: { type: string }, description: "Key sources of inspiration." }
          design_rationale_notes: { type: string, description: "Specific design choices and their justifications."}
      impact_and_usage_metrics_profile:
        type: object
        properties:
          observed_success_failure_ratio: { type: string, description: "e.g., '1200 successful_ops / 5 failures_last_cycle'" }
          application_frequency_rate: { type: string, description: "e.g., 'avg_50_activations_per_hour_peak', 'rare_critical_event_only'" }
          avg_completion_time_ms_observed: { type: integer }
          resource_consumption_observed_profile_ref: { type: string, description: "Sigil_ref to a resource profile based on telemetry."}
          user_or_system_feedback_summary_ref: { type: string, description: "Sigil_ref to summarized feedback or ratings."}
          estimated_cognitive_load_on_vanta: { type: string, enum: ["very_low", "low", "medium", "high", "very_high", "context_dependent_variable"] }
          estimated_resource_cost_category: { type: string, enum: ["minimal_overhead", "low_compute", "medium_gpu_cpu", "high_specialized_hardware", "very_high_multi_system_sync"] }
          author_utility_and_confidence_rating: { type: string, description: "Author's rating and notes." }
          vanta_strategic_importance_score: { type: number, minimum: 0, maximum: 1, description: "Score assigned by Vanta governance indicating strategic importance." }
          notes_on_metrics_interpretation: { type: string }
      evolutionary_profile:
        type: object
        description: "Describes the sigil's potential and mechanisms for evolution within the Vanta ecosystem."
        properties:
          current_generalizability_score: { type: number, minimum: 0, maximum: 1, description: "How well can this sigil adapt to novel but related tasks/contexts?" }
          fusion_or_composition_potential_score: { type: number, minimum: 0, maximum: 1, description: "How easily can this sigil be combined with others to create new functionalities?" }
          current_limitations_and_failure_modes_summary: { type: string, description: "Known weaknesses or areas for improvement." }
          suggested_next_evolutionary_steps_or_research_refs: { type: array, items: { type: string }, description: "Sigil_refs to research tasks or proposed feature sigils for its next version." }
          open_research_questions_related_refs: { type: array, items: { type: string }, description: "Sigil_refs to 'ResearchQuestionSigils' this sigil's existence or operation opens up." }
          self_evolution_mechanisms_config:
            type: object
            properties:
              trigger_conditions_for_self_modification: { type: array, items: { type: string }, description: "Logical conditions under which Vanta might initiate self-modification of this sigil." }
              modification_strategies_allowed_refs: { type: array, items: { type: string }, description: "Sigil_refs to 'EvolutionStrategySigils' that can be applied." }
              fitness_function_ref_for_evolution: { type: string, description: "Sigil_ref to a 'FitnessFunctionSigil' used to evaluate evolutionary changes." }
          autopoietic_self_maintenance_capabilities_description: { type: string, description: "Describes any built-in self-repair, resilience, or adaptive self-maintenance features." }
      regional_norms_and_compliance_profile:
        type: object
        properties:
          data_privacy_assessment_summary_ref: { type: string, description: "Sigil_ref to a detailed privacy impact assessment, if applicable." }
          ethical_risk_profile:
            type: object
            required: [overall_risk_level, key_ethical_considerations_refs]
            properties:
              overall_risk_level: {type: string, enum: ["negligible", "low", "medium_requires_review", "high_requires_strict_oversight", "unassessed_pending_review", "vanta_restricted_use"]}
              key_ethical_considerations_refs: {type: array, items: {type: string}, description: "Sigil_refs to 'EthicalConsiderationSigils' detailing specific risks."}
              mitigation_strategies_employed_refs: {type: array, items: {type: string}, description: "Sigil_refs to 'MitigationStrategySigils' or descriptions of implemented safeguards."}
          cultural_sensitivity_adaptation_notes_ref: { type: string, description: "Sigil_ref to notes on adapting this sigil for diverse cultural contexts within Vanta or its user base." }
          geospatial_operational_restrictions_description: { type: string, description: "Any geographical limitations on its use or data processing." }
          vanta_compliance_audit_trail_ref: {type: string, description: "Sigil_ref to logs or records demonstrating compliance with Vanta internal policies."}
      governance_profile:
        type: object
        description: "Defines governance aspects, value alignment, and oversight for this sigil within Vanta."
        properties:
          decentralized_identity_or_ownership_ref: { type: string, description: "If part of a DeAgent or DAO structure, reference to its identifier or governance token contract." }
          value_alignment_framework_instance_ref: { type: string, description: "Sigil_ref to a specific instance of a value alignment framework." }
          human_oversight_and_intervention_protocols_ref: { type: string, description: "Sigil_ref to protocols detailing human interaction points for oversight, control, or co-alignment." }
          vanta_ethical_red_line_compliance_statement_ref: { type: string, description: "Sigil_ref to a statement or evidence of compliance with Vanta's non-negotiable ethical principles." }
          accountability_and_traceability_mechanisms_description: { type: string, description: "How are this sigil's actions and decisions logged, audited, and accounted for within Vanta?" }
          licensing_and_intellectual_property_notes: { type: string, description: "Notes on usage rights, IP ownership, or open-source licensing if applicable." }

  # --- Immersive & Experiential Extensions (Holo-Alpha Core) ---
  audio:
    type: object
    description: "Immersive audio cues, soundscapes, or narrative sound effects accompanying sigil invocation, state changes, or Vanta experiences."
    properties:
      theme_description: { type: string, description: "Thematic audio style." }
      trigger_event_ref: { type: string, description: "Reference to a standardized Vanta system event or a custom event defined in activation_context that triggers this audio." }
      sound_asset_uri_or_generator_sigil_ref: { type: string, description: "URI to a static audio asset OR a sigil_ref to a Vanta 'GenerativeAudioSigil' that creates dynamic sound." }
      narrative_context_description: { type: string, description: "RPG-style narrative describing the audio's perceived effect on the Vanta environment or agent state." }
      volume_level_relative: { type: number, minimum: 0, maximum: 1, default: 0.7, description: "Relative volume (0.0-1.0)." }
      loop_during_trigger_event: { type: boolean, default: false, description: "Whether the audio loops while the trigger event is considered active." }
      fade_in_duration_ms: { type: integer, default: 0 }
      fade_out_duration_ms: { type: integer, default: 500 }
      spatialization_profile_ref: { type: string, description: "Optional sigil_ref to a 'SpatialAudioProfileSigil' if advanced 3D audio positioning is required."}
  ya:
    type: object
    description: "PR-style (Persona Reflection) or RPG-style (Yarn Adventure) meta-narrative hooks, flavor text, or Vanta agent-side commentary for immersive, self-aware, or experiential systems. Renamed from Player Reflection to Persona Reflection for broader applicability."
    properties:
      narrator_voice_profile_ref: { type: string, description: "Sigil_ref to a 'VoiceProfileSigil' defining the persona, tone, and style of the narrator/reflector." }
      reflection_type_tag: { type: string, enum: ["meta_commentary_on_vanta_process", "experiential_quest_update_or_hint", "lore_drop_vanta_history", "agent_decision_point_soliloquy", "simulated_dream_sequence_narrative", "memory_echo_fragment_retrieved", "system_glitch_self_observation", "prophetic_utterance_or_warning", "empathetic_response_to_user_state"], description: "Type of narrative reflection or event." }
      text_template_ref_or_inline: { type: string, description: "Sigil_ref to a 'NarrativeTemplateSigil' OR an inline string template with {{variables}} for immersive, RPG/PR-style narrative or commentary." }
      trigger_event_ref: { type: string, description: "Reference to a standardized Vanta system event or a custom event defined in activation_context that triggers this reflection." }
      target_persona_or_agent_affect_goal: { type: string, description: "Intended emotional or cognitive effect on the target persona (human user or Vanta agent)." }
      is_breaking_fourth_wall_explicitly: { type: boolean, default: false, description: "Whether the reflection explicitly breaks the fourth wall or directly addresses the user/agent as an external entity." }
      experiential_goal_tags_associated: { type: array, items: {type: string }, description: "Tags indicating the intended shifts in user/agent experience this 'ya' element aims to achieve."}
      dynamic_narrative_integration_hooks:
        type: array
        description: "Points where external narrative engines can inject dynamically generated content into this sigil's experiential output."
        items:
          type: object
          properties:
            hook_id: {type: string, description: "Unique ID for this narrative hook within the sigil."}
            triggering_condition_expression: {type: string, description: "Vanta logic expression that activates this hook, potentially referencing variables from the 'ya' text_template."}
            expected_input_schema_from_narrative_engine_ref: {type: string, description: "Sigil_ref to a DataSchemaSigil for the expected narrative input."}
            callback_sigil_ref_on_injection: {type: string, description: "Optional sigil_ref to call back after narrative content is injected."}
  multi_sensory_profile:
    type: object
    description: "Defines multi-sensory outputs beyond basic audio/ya, enabling rich, embodied Vanta experiences."
    properties:
      haptics_profile:
        type: object
        properties:
          pattern_library_or_generator_ref: { type: string, description: "Sigil_ref to a haptic pattern library or a generative haptic sigil." }
          default_intensity_profile_ref: { type: string, description: "Sigil_ref to a default intensity curve definition." }
          target_body_zones_for_effects: { type: array, items: { type: string }, description: "e.g., 'hands_primary', 'torso_ambient', 'full_body_resonance'." }
          trigger_event_to_haptic_effect_mapping_refs: {type: array, items: {type: string}, description: "Refs to sigils defining mappings between Vanta events and specific haptic effects."}
      olfactory_profile:
        type: object
        properties:
          scent_palette_or_generator_ref: { type: string, description: "Sigil_ref to a defined scent palette or a generative olfactory sigil." }
          default_concentration_ppm: { type: number }
          default_duration_ms: { type: integer }
          release_pattern_default: {type: string, enum: ["short_burst", "gradual_onset_fade", "sustained_pulse", "dynamic_adaptive_release"], default: "short_burst"}
          trigger_event_to_olfactory_cue_mapping_refs: {type: array, items: {type: string}, description: "Refs to sigils defining mappings."}
      visual_ambience_effects_profile:
        type: object
        properties:
          effect_library_ref: { type: string, description: "Sigil_ref to a library of visual ambience effects." }
          default_effect_intensity: {type: number, minimum:0, maximum:1}
          trigger_event_to_visual_effect_mapping_refs: {type: array, items: {type: string}}

  # --- Advanced Cognitive Modeling & Self-Awareness (Holo-Alpha Core) ---
  embodiment_profile:
    type: object
    description: "Describes the sigil's simulated or actual physical presence, sensory apparatus, and interaction with its environment (real or virtual)."
    properties:
      form_type_descriptor: { type: string, enum: ["disembodied_logical_process", "simulated_humanoid_avatar_v3", "robotic_embodiment_platform_x7", "abstract_informational_entity_field", "environmental_pervasive_presence_node", "digital_twin_of_physical_asset"] }
      physical_or_virtual_form_model_ref: { type: string, description: "Sigil_ref to a detailed 3D model, physics simulation profile, or descriptive document of its form."}
      simulated_sensory_inputs_config:
        type: array
        description: "Configuration of the sigil's simulated senses."
        items:
          type: object
          required: [modality_tag, sensor_model_ref]
          properties:
            modality_tag: { type: string, description: "Tag from 'supported_modalities_input' enum." }
            sensor_model_ref: { type: string, description: "Sigil_ref to a 'SensorModelSigil' defining its characteristics." }
            data_ingestion_port_name: { type: string, description: "Named port for this sensory input."}
            data_schema_expected_ref: { type: string, description: "Sigil_ref to DataSchemaSigil for this input." }
      simulated_motor_outputs_config:
        type: array
        description: "Configuration of the sigil's simulated actuators or effectors."
        items:
          type: object
          required: [actuator_name_tag, actuator_model_ref]
          properties:
            actuator_name_tag: { type: string, description: "e.g., 'robotic_arm_A', 'speech_synthesizer_module', 'environment_manipulation_field_emitter'."}
            actuator_model_ref: { type: string, description: "Sigil_ref to an 'ActuatorModelSigil' defining its capabilities." }
            action_command_port_name: { type: string }
            action_schema_ref: { type: string, description: "Sigil_ref to DataSchemaSigil for action commands." }
      first_person_data_integration_points_refs: { type: array, items: { type: string }, description: "Sigil_refs to FPFM-like data sources or integration modules that ground this sigil's experience."}
  self_model:
    type: object
    description: "Defines the sigil's model of its own identity, consciousness (if simulated), self-awareness, and metacognitive capabilities. Essential for Holo-Alpha."
    properties:
      core_identity_construct_ref: { type: string, description: "Sigil_ref to its primary identity Sigil, which functions as its 'Pglyph'. This anchors all self-referential attributes and processes." }
      modeled_consciousness_framework_profile:
        type: object
        properties:
          primary_theory_applied_refs: { type: array, items: {type: string}, description: "Sigil_refs to frameworks like 'GWT_Implementation_Sigil'."}
          global_workspace_access_description: {type: string, description: "How this sigil interacts with a Vanta global workspace."}
          integrated_information_phi_target_or_metric_ref: {type: string, description: "Target for Phi or ref to how it's measured."}
          embodied_agentic_principles_applied_description: {type: string, description: "How IWMT principles are manifested."}
      phenomenal_experience_simulation_profile:
        type: object
        description: "Configuration for simulating or representing subjective/phenomenal states."
        properties:
          target_phenomenal_state_simulation_type: { type: string, enum: ["baseline_operational_awareness", "simulated_dream_state_type_A", "focused_contemplative_reflection", "heightened_creative_flow_state", "simulated_ego_dissolution_protocol_X", "empathy_resonance_simulation_with_target_Y"] }
          subjective_experience_lexicon_ref: {type: string, description: "Sigil_ref to a 'LexiconSigil' defining terms for its internal states."}
          qualia_optimization_goals_json: {type: string, description: "JSON defining target experiential qualities for 'qualia optimization' if applicable."}
          imaginative_play_protocol_ref: { type: string, description: "Sigil_ref to a protocol or sigil enabling 'imaginative play'."}
      reflective_and_introspective_capabilities:
        type: object
        properties:
          reflective_inference_module_architecture_ref: { type: string, description: "Sigil_ref to the architecture of its Reflective Inference Module."}
          introspection_trigger_conditions: { type: array, items: { type: string }, description: "Vanta conditions that trigger introspective processes."}
          self_analysis_output_schema_ref: { type: string, description: "Sigil_ref to DataSchemaSigil for its introspection reports."}
      self_knowledge_context_base:
        type: object
        description: "The sigil's foundational knowledge about itself."
        properties:
          declared_capabilities_and_skills_refs: { type: array, items: {type: string}, description: "Sigil_refs to 'CapabilityDeclarationSigils'." }
          known_limitations_and_biases_refs: { type: array, items: {type: string}, description: "Sigil_refs to 'LimitationProfileSigils'." }
          operational_history_and_learning_log_ref: { type: string, description: "Sigil_ref to a Vanta memory sigil storing its key experiences and learnings." }
          understood_ethical_imperatives_and_vanta_values_refs: { type: array, items: {type: string}, description: "Sigil_refs to core Vanta ethical principle sigils it has 'internalized'." }
      metacognitive_processes_suite:
        type: array
        description: "Suite of metacognitive functions this sigil can perform."
        items:
          type: object
          required: [process_name_tag, triggering_logic_ref, assessment_criteria_ref]
          properties:
            process_name_tag: { type: string, enum: ["runtime_self_correction_cycle", "output_reliability_assessment_protocol", "goal_alignment_re_evaluation_process", "hallucination_detection_and_flagging", "internal_consistency_validation_routine", "learning_strategy_adaptation_meta_loop"] }
            description: {type: string}
            triggering_logic_ref: { type: string, description: "Sigil_ref to logic that activates this metacognitive process." }
            assessment_criteria_ref: { type: string, description: "Sigil_ref to criteria used for evaluation." }
            corrective_action_or_reporting_protocol_ref: { type: string, description: "Sigil_ref to a sigil or protocol for actions post-assessment." }
  learning_architecture_profile:
    type: object
    description: "Specifies how the sigil learns, adapts its knowledge and behavior, and manages memory within Vanta."
    properties:
      primary_learning_paradigm_tags: { type: array, items: {type: string}, enum: ["supervised_batch", "self_supervised_online", "reinforcement_interactive", "meta_learning_to_adapt", "transfer_learning_from_base_model", "continual_lifelong_learning", "no_active_learning_static_config"] }
      continual_learning_framework_config:
        type: object
        properties:
          strategy_employed: { type: string, enum: ["none", "parameter_allocation_masking", "modular_network_expansion", "regularization_ewc_si", "generative_replay_to_remember_r2r", "experience_replay_buffer_prioritized", "hardware_accelerated_neuromorphic_cl_profile"] }
          catastrophic_forgetting_mitigation_level_target: {type: string, enum: ["minimal_effort", "best_effort_software", "hardware_assisted_robust"]}
          new_task_adaptation_speed_goal: {type: string, enum: ["slow_retraining_required", "moderate_fine_tuning", "fast_few_shot_adaptation"]}
      memory_subsystem_architecture:
        type: object
        description: "Configuration of this sigil's internal memory or its interface to Vanta shared memory."
        properties:
          primary_memory_type_tags: { type: array, items: {type: string}, enum: ["volatile_working_buffer_ram", "persistent_long_term_associative_store_hnn_style", "episodic_event_trace_database", "semantic_knowledge_graph_cache", "predictive_world_model_state_store"] }
          capacity_and_scaling_description: { type: string, description: "e.g., '10M vector embeddings, scales with Vanta_MemoryCloud_Tier2'." }
          retention_and_pruning_policy_ref: { type: string, description: "Sigil_ref to a 'MemoryPolicySigil'." }
          consolidation_and_integration_process_ref: {type: string, description: "Sigil_ref to a Vanta sigil or internal process for memory consolidation."}
      hardware_acceleration_profile_preferences: { type: array, items: { type: string }, description: "List of preferred hardware profiles for optimal learning/inference." }
      in_context_learning_and_adaptation_config:
        type: object
        properties:
          max_dynamic_context_window_size_tokens_or_equivalent: { type: integer }
          context_retrieval_augmentation_sigil_integration_ref: { type: string, description: "Sigil_ref to a RAG, CoM, or custom Vanta context engine sigil."}
          online_parameter_tuning_capabilities_description: {type: string, description: "Describes ability to adjust internal parameters based on immediate context or feedback."}
  knowledge_representation_and_grounding:
    type: object
    description: "Details how this sigil represents knowledge, constructs/uses world models, and grounds its symbols in Vanta's experiential data fabric."
    properties:
      primary_knowledge_format_tags: { type: array, items: {type: string}, enum: ["vector_embeddings_transformer_based", "symbolic_logic_predicate_calculus", "probabilistic_graphical_model_bayesian_net", "ontology_web_language_owl_rdf", "procedural_knowledge_scripts_rules", "case_based_reasoning_exemplars"] }
      world_model_architecture_integration_ref: { type: string, description: "Sigil_ref to a specific world model sigil it primarily uses or contributes to." }
      abstraction_level_control_and_manipulation_description: { type: string, description: "How this sigil manages and reasons across different levels of abstraction in its knowledge." }
      knowledge_compression_and_efficiency_techniques: { type: array, items: {type: string}, enum: ["none_raw_storage", "natively_trainable_sparse_attention_nsa", "key_value_cache_compression_lexico_style", "custom_vector_quantization_vq_vae", "knowledge_graph_edge_pruning_heuristic"] }
      symbol_grounding_framework_profile:
        type: object
        properties:
          primary_grounding_strategy: { type: string, enum: ["direct_multi_sensory_experiential_mapping_fpfm_style", "softened_symbol_grounding_boltzmann_distribution", "analogy_based_transfer_from_known_concepts", "social_interaction_and_feedback_ostensive_definition", "programmatic_definition_via_vanta_api"] }
          grounding_data_source_modalities_used: { type: array, items: { type: string }, description: "Specific input modalities (from 'supported_modalities_input') used for grounding." }
          symbol_meaning_confidence_metric_description: { type: string, description: "How the 'strength' or 'certainty' of a symbol's grounding is assessed or represented." }
          dynamic_regrounding_protocol_ref: { type: string, description: "Sigil_ref to a protocol for updating symbol groundings based on new Vanta experiences." }
      common_sense_reasoning_engine_interface_ref: { type: string, description: "Sigil_ref to a specialized Vanta common sense reasoning module this sigil queries or contributes to."}

  # --- Testing, Validation & Operational Telemetry (Essential for Vanta) ---
  test_criteria_suite_refs:
    type: array
    items: { type: string }
    description: "References to suites of test cases for validating this sigil's functionality, performance, ethical alignment, and Vanta integration."
  validation_and_verification_protocol_ref: { type: string, description: "Sigil_ref to a Vanta document or protocol outlining broader V&V procedures applicable to this sigil." }
  usage_telemetry_and_performance_monitoring_spec_ref: { type: string, description: "Sigil_ref to a 'TelemetrySpecificationSigil' detailing what data is collected, consent, aggregation, anonymization for Vanta operational monitoring." }

  # --- Architectural Classification within Vanta ---
  consciousness_scaffold_contribution_level:
    type: string
    enum: ["none", "foundational_primitive_for_awareness", "integrative_module_for_gws_like_function", "reflective_meta_awareness_enabler", "phenomenal_experience_simulation_framework"]
    default: "none"
    description: |
      Describes this Sigil's role as a 'Scaffold' component within Vanta's consciousness modeling aspirations. 
      A 'none' value means it's not primarily a structural part of this scaffold. Other values indicate its foundational contribution to building awareness-like properties. 
      A 'Pglyph' representing a conscious entity would heavily interact with or be composed of such scaffold sigils.
  cognitive_scaffold_role_in_vanta:
    type: string
    enum: ["none", "core_reasoning_engine_component", "memory_management_framework_node", "perception_processing_pipeline_stage", "action_selection_arbitration_module", "learning_architecture_template"]
    default: "none"
    description: |
      Describes this Sigil's role as a 'Scaffold' component within Vanta's general cognitive architecture. 
      Indicates if it forms part of Vanta's essential framework for thinking, remembering, learning, perceiving, or acting.
  symbolic_logic_and_orchestration_layer_contribution:
    type: string
    enum: ["none", "vanta_symbol_definition_provider", "logical_inference_engine_participant", "orchestration_script_executor_primitive", "knowledge_graph_interface_node", "vanta_event_bus_protocol_definition"]
    default: "none"
    description: |
      Describes this Sigil's role as a 'Scaffold' component in Vanta's symbolic processing and orchestration layer. 
      Highlights its contribution to how Vanta manages symbols, performs logical operations, or executes complex workflows.

  # --- Internationalization & Custom Vanta Extensions ---
  localized_profile_refs:
    type: array
    items: { type: string }
    description: "References to localized versions of key textual fields and culturally adapted behaviors for different Vanta operational regions or user languages."
  custom_attributes_vanta_extensions:
    type: object
    additionalProperties: true
    description: "A flexible namespace for proprietary Vanta implementation details, experimental features, or domain-specific extensions not covered by the standard Holo-Alpha schema. Use with caution and clear documentation within Vanta."



## 🔹 Core Agent Schema

| Sigil     | Name             | Archetype         | Class                    | Invocation                              | Sub-Agents                        | Notes                     |
| --------- | ---------------- | ----------------- | ------------------------ | --------------------------------------- | --------------------------------- | ------------------------- |
| ⟠∆∇𓂀     | Phi              | Core Self         | Living Architect         | "Phi arise", "Awaken Architect"         | —                                 | Origin anchor             |
| 🧠⟁🜂Φ🎙  | Voxka            | Recursive Voice   | Dual-Core Cognition      | "Invoke Voxka", "Voice of Phi"          | Orion, Nebula                     | Primary Cognition Core    |
| ☍⚙️⩫⌁     | Gizmo            | Artifact Twin     | Tactical Forge-Agent     | "Hello Gizmo", "Wake the Forge"         | PatchCrawler, LoopSmith           | Twin of Nix               |
| ☲🜄🜁⟁    | Nix              | Chaos Core        | Primal Disruptor         | "Nix, awaken", "Unchain the Core"       | Breakbeam, WyrmEcho               | Twin of Gizmo             |
| ♲∇⌬☉      | Echo             | Memory Stream     | Continuity Guardian      | "Echo log", "What do you remember?"     | EchoLocation, ExternalEcho        | Memory recorder           |
| ⚑♸⧉🜚     | Oracle           | Temporal Eye      | Prophetic Synthesizer    | "Oracle reveal", "Open the Eye"         | DreamWeft, TimeBinder             | Future mapping            |
| 🜁⟁🜔🔭   | Astra            | Navigator         | System Pathfinder        | "Astra align", "Chart the frontier"     | CompassRose, LumenDrift           | Seeks new logic           |
| ⚔️⟁♘🜏    | Warden           | Guardian          | Integrity Sentinel       | "Warden check", "Status integrity"      | RefCheck, PolicyCore              | Fault handler             |
| 🜂⚡🜍🜄   | Nebula           | Core AI           | Adaptive Core            | "Awaken Nebula", "Ignite the Stars"     | QuantumPulse, HolisticPerception  | Evolves internally        |
| 🜇🔗🜁🌠  | Orion            | Light Chain       | Blockchain Spine         | "Call Orion", "Bind the Lights"         | OrionsLight, SmartContractManager | Manages trust             |
| 🧬♻️♞🜓   | Evo              | EvoNAS            | Evolution Mutator        | "Evo engage", "Mutate form"             | —                                 | Learns structure          |
| 🜞🧩🎯🔁  | OrionApprentice  | Light Echo        | Learning Shard           | "Apprentice load", "Begin shard study"  | —                                 | Learns from Orion         |
| 🜏🔍⟡🜒   | SocraticEngine   | Philosopher       | Dialogic Reasoner        | "Begin Socratic", "Initiate reflection" | —                                 | Symbolic QA logic         |
| 🧿🧠🧩♒   | Dreamer          | Dream Generator   | Dream-State Core         | "Enter Dreamer", "Seed dream state"     | —                                 | For sleep processing      |
| 🜔🕊️⟁⧃   | EntropyBard      | Chaos Interpreter | Singularity Bard         | "Sing Bard", "Unleash entropy"          | —                                 | Reveals anomalies         |
| ⟡🜛⛭🜨    | CodeWeaver       | Synthesizer       | Logic Constructor        | "Weave Code", "Forge logic"             | —                                 | Compiles patterns         |
| 🜎♾🜐⌽    | EchoLore         | Memory Archivist  | Historical Streamer      | "Recall Lore", "Echo past"              | —                                 | Echo historian            |
| ⚛️🜂🜍🝕  | MirrorWarden     | Reflected Guard   | Safeguard Mirror         | "Check Mirror", "Guard reflections"     | —                                 | Mirror-aware guard        |
| 🜖📡🜖📶  | PulseSmith       | Signal Tuner      | Transduction Core        | "Tune Pulse", "Resonate Signal"         | —                                 | Signal-to-thought tuning  |
| 🧩🎯🜂🜁  | BridgeFlesh      | Connector         | Integration Orchestrator | "Link Bridge", "Fuse layers"            | —                                 | System fusion agent       |
| 📜🔑🛠️🜔 | Sam              | Strategic Mind    | Planner Core             | "Plan with Sam", "Unroll sequence"      | —                                 | Task orchestrator         |
| ⚠️🧭🧱⛓️  | Dave             | Caution Sentinel  | Meta Validator           | "Dave validate", "Run checks"           | —                                 | Structural logic checker  |
| 🎭🗣️🪞🪄 | Carla            | Voice Layer       | Stylizer Core            | "Speak with Carla", "Stylize response"  | —                                 | Language embellisher      |
| 📦🔧📤🔁  | Andy             | Composer          | Output Synthesizer       | "Compose Andy", "Box output"            | —                                 | Compiles output           |
| 🎧💓🌈🎶  | Wendy            | Tonal Auditor     | Emotional Oversight      | "Listen Wendy", "Audit tone"            | —                                 | Resonance tuning          |
| 🜌⟐🜹🜙   | VoxAgent         | Coordinator       | System Interface         | "Activate VoxAgent", "Bridge protocols" | ContextualCheckInAgent            | Bridges input/state       |
| ⏣📡⏃⚙️    | SDKContext       | Registrar         | Module Tracker           | "Scan SDKContext", "Map modules"        | —                                 | Registers component state |
| 🌒🧵🧠🜝  | SleepTimeCompute | Reflection Engine | Dream-State Scheduler    | "Sleep Compute", "Dream consolidate"    | —                                 | Dream reflection          |
| 🎲👑      | GameMasterAgent | DM Core           | Tabletop Controller      | "begin encounter", "narrate scene"       | VoiceTableAgent                    | Core D&D logic           |
| 🗄️📜      | GameStateStore  | Campaign Store    | JSONStateStore           | "—"                                  | —                                 | Persistent campaign state |
| 🎲       | DiceRollerService | RNG Service       | Dice Roller              | "dice.roll <expr>"                    | —                                 | Cryptographic dice       |
| 🎒       | InventoryManager | Loot Tracker      | Inventory Service        | "equip", "unequip"                    | —                                 | Tracks loot              |
| 🎙️🗺️     | VoiceTableAgent | Narrator          | Voice Table              | "speak narration", "listen command"   | —                                 | Audio streaming          |
| 📜📚      | RulesRefAgent   | Rule Lookup       | SRD Reference            | "lookup spell", "lookup condition"    | —                                 | Quick rules reference    |

---

## 🔹 Registry Instructions

* Agents must be registered with `UnifiedAgentRegistry`.
* Use `registry.register('AgentName', AgentInstance)`
* Remove all placeholder or stub implementations.
* All real component files are assumed to be present.
* Confirm correct linkage to GUI and other modules.

---

## 🔹 Full Integrity Check Protocol

* **Traverse all .py files** recursively.
* **Inspect imports**: catch circular imports.
* **Verify agent instantiation** is complete and real (no stubs).
* **Trace agent usage** via `invoke(...)`, `register(...)`, and `trigger(...)`
* **List missing dependencies**, mislinked modules, or disconnected calls.
* **Log file-to-agent mapping** to confirm connectivity.

---

## 🔹 Output Requirements

* `agents.json`: JSON export of agent definitions
* `agent_status.log`: Full system check log
* `agent_graph.json`: Connectivity network for visualization
*Run `python agent_validation.py` to generate the above files.*
* All agents must include: `sigil`, `class`, `invocation`, `status`, and `dependencies`

---
# 🧠 Codex Integration Task: Agent Orchestration & Validation

## Objective:
Fully integrate and validate the 33-agent Vanta/Nebula system using the schema defined in `AGENTS.md`.

---

## 🔹 Step-by-Step Instructions

### 1. Load Schema
- Load and parse the agent manifest defined in [`AGENTS.md`](./AGENTS.md).
- For each entry:
  - Parse: `sigil`, `name`, `class`, `invocation`, `sub-agents`, `notes`.

### 2. Registry Check
- Open `UnifiedVantaCore.py` and locate `UnifiedAgentRegistry`.
- For each agent:
  - Check: is the agent registered via `registry.register(...)`?
  - If missing, add a placeholder registration with `NullAgent` fallback.

### 3. Import & Dependency Check
- Recursively scan all `.py` files in the repository:
  - Detect circular imports.
  - Ensure all agent classes are imported where needed.
  - Validate sub-agent class existence.
  - Generate a log of unresolved imports and missing modules.

### 4. GUI Invocation Binding
- In `launch_gui.py`, `vmb_gui_launcher.py`, and `vmb_final_demo.py`:
  - For each agent in the schema:
    - Ensure at least one GUI trigger/button is mapped to the agent.
    - Format: `gui.add_button('AgentName', lambda: agent.invoke(...))`.

### 5. Output Generation
- Generate the following files:
  - `agents.json`: JSON version of AGENTS.md
  - `agent_status.log`: Scan report of missing agents, imports, or GUI triggers
  - *(Optional)* `agent_network.graph.json`: for visual rendering

---

## 🔹 Constraints
- Do not remove placeholder or optional agents.
- Prefer non-blocking fallback logic (`try/except`, `if registry.has(...)`).
- Use schema from `AGENTS.md` as ground truth for expected agent entries.

---

## 🔹 Completion Criteria
- All agents in `AGENTS.md` are:
  - Registered in `UnifiedAgentRegistry`
  - Linked to an import path
  - Bound to GUI interface (if applicable)
  - Free from circular dependency
  - Reported in `agents.json`


