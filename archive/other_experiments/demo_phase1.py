"""Quick demo of Phase 1 working."""

from voxsigil_memory import build_context

result = build_context('What is machine learning?', budget_tokens=512, mode='balanced')

print('=== VoxSigil VME Phase 1 Demo ===')
print('Query:', result.query)
print('Budget:', result.budget_tokens, 'tokens')
print('Mode:', result.mode)
print('Version:', result.version)
print('Signature:', result.signature)
print('Compressed bytes:', len(result.compressed_content))
print('Metadata keys:', list(result.metadata.keys()))
print()
print('✓ Phase 1 Fully Functional!')
