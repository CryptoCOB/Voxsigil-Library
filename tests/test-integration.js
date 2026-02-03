/**
 * Integration Tests for VoxSigil Library (JavaScript)
 */

const fs = require('fs');
const path = require('path');

// Test counter
let testsRun = 0;
let testsPassed = 0;
let testsFailed = 0;

// Test helper
function test(name, fn) {
  testsRun++;
  try {
    fn();
    testsPassed++;
    console.log(`✓ ${name}`);
  } catch (error) {
    testsFailed++;
    console.log(`✗ ${name}`);
    console.log(`  Error: ${error.message}`);
  }
}

// Assertion helper
function assert(condition, message) {
  if (!condition) {
    throw new Error(message || 'Assertion failed');
  }
}

console.log('Running VoxSigil Library Integration Tests (JavaScript)');
console.log('='.repeat(70));
console.log();

// Test 1: Module loads
test('Module loads successfully', () => {
  const voxsigil = require('../src/index');
  assert(voxsigil !== undefined, 'Module should load');
  assert(typeof voxsigil.loadAgentConfig === 'function', 'loadAgentConfig should be a function');
  assert(typeof voxsigil.computeChecksum === 'function', 'computeChecksum should be a function');
  assert(typeof voxsigil.getMetadata === 'function', 'getMetadata should be a function');
});

// Test 2: Get metadata
test('getMetadata returns valid data', () => {
  const voxsigil = require('../src/index');
  const metadata = voxsigil.getMetadata();
  
  assert(metadata.name === '@voxsigil/library', 'Name should match');
  assert(metadata.version === '1.0.0', 'Version should be 1.0.0');
  assert(Array.isArray(metadata.capabilities), 'Capabilities should be an array');
  assert(metadata.capabilities.length > 0, 'Should have capabilities');
});

// Test 3: Compute checksum
test('computeChecksum works correctly', () => {
  const voxsigil = require('../src/index');
  const testData = 'Hello, VoxSigil!';
  const checksum = voxsigil.computeChecksum(testData);
  
  assert(typeof checksum === 'string', 'Checksum should be a string');
  assert(checksum.length === 64, 'SHA256 should be 64 hex characters');
  assert(/^[a-f0-9]+$/.test(checksum), 'Checksum should be hex');
});

// Test 4: Agent files exist
test('Agent files exist', () => {
  const agentsDir = path.join(__dirname, '..', 'src', 'agents');
  const files = ['boot.md', 'agents.md', 'memory.md', 'hooks-config.json'];
  
  for (const file of files) {
    const filePath = path.join(agentsDir, file);
    assert(fs.existsSync(filePath), `${file} should exist`);
  }
});

// Test 5: Load agent config
test('loadAgentConfig loads all files', () => {
  const voxsigil = require('../src/index');
  const config = voxsigil.loadAgentConfig();
  
  assert(config.boot, 'Should have boot content');
  assert(config.agents, 'Should have agents content');
  assert(config.memory, 'Should have memory content');
  assert(config.hooks, 'Should have hooks config');
  
  assert(typeof config.boot === 'string', 'boot should be a string');
  assert(typeof config.agents === 'string', 'agents should be a string');
  assert(typeof config.memory === 'string', 'memory should be a string');
  assert(typeof config.hooks === 'object', 'hooks should be an object');
});

// Test 6: Hooks config is valid JSON
test('hooks-config.json is valid', () => {
  const voxsigil = require('../src/index');
  const config = voxsigil.loadAgentConfig();
  
  assert(config.hooks.hooks, 'Should have hooks section');
  assert(typeof config.hooks.hooks === 'object', 'hooks should be an object');
  
  // Check a specific hook
  const bootHook = config.hooks.hooks['boot-md'];
  assert(bootHook, 'boot-md hook should exist');
  assert(bootHook.enabled === true, 'boot-md should be enabled');
  assert(bootHook.trigger === 'on_startup', 'boot-md should trigger on startup');
});

// Test 7: Checksum utilities
test('Checksum utilities work', () => {
  const checksum = require('../src/utils/checksum');
  const agentsDir = path.join(__dirname, '..', 'src', 'agents');
  
  const checksums = checksum.computeAgentChecksums(agentsDir);
  
  assert(checksums['boot.md'], 'Should have boot.md checksum');
  assert(checksums['agents.md'], 'Should have agents.md checksum');
  assert(checksums['memory.md'], 'Should have memory.md checksum');
  assert(checksums['hooks-config.json'], 'Should have hooks-config.json checksum');
  
  // All checksums should be 64 character hex strings
  for (const [file, hash] of Object.entries(checksums)) {
    assert(hash.length === 64, `${file} checksum should be 64 characters`);
    assert(/^[a-f0-9]+$/.test(hash), `${file} checksum should be hex`);
  }
});

// Test 8: File checksums can be verified
test('File checksum verification works', () => {
  const voxsigil = require('../src/index');
  const agentsDir = path.join(__dirname, '..', 'src', 'agents');
  const bootPath = path.join(agentsDir, 'boot.md');
  
  // Compute checksum
  const expectedChecksum = voxsigil.computeFileChecksum(bootPath);
  
  // Verify it matches
  const isValid = voxsigil.verifyFileChecksum(bootPath, expectedChecksum);
  assert(isValid, 'File checksum should verify correctly');
  
  // Verify it fails with wrong checksum
  const isInvalid = voxsigil.verifyFileChecksum(bootPath, 'wrong_checksum');
  assert(!isInvalid, 'Should fail with wrong checksum');
});

// Test 9: Agent files have content
test('Agent files have substantial content', () => {
  const voxsigil = require('../src/index');
  const config = voxsigil.loadAgentConfig();
  
  // BOOT.md should be 250-300+ lines
  assert(config.boot.length > 5000, 'BOOT.md should have substantial content');
  
  // AGENTS.md should be 300-350+ lines  
  assert(config.agents.length > 8000, 'AGENTS.md should have substantial content');
  
  // MEMORY.md should be 250-300+ lines
  assert(config.memory.length > 8000, 'MEMORY.md should have substantial content');
});

// Test 10: Package.json exists and is valid
test('package.json is valid', () => {
  const packagePath = path.join(__dirname, '..', 'package.json');
  assert(fs.existsSync(packagePath), 'package.json should exist');
  
  const pkg = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
  
  assert(pkg.name === '@voxsigil/library', 'Package name should match');
  assert(pkg.version === '1.0.0', 'Version should be 1.0.0');
  assert(Array.isArray(pkg.keywords), 'Should have keywords');
  assert(pkg.keywords.includes('molt-agent'), 'Should have molt-agent keyword');
  assert(pkg.repository.url === 'https://github.com/CryptoCOB/Voxsigil-Library', 'Repository URL should match');
});

// Summary
console.log();
console.log('='.repeat(70));
console.log('Test Summary');
console.log('='.repeat(70));
console.log(`Total: ${testsRun}`);
console.log(`Passed: ${testsPassed}`);
console.log(`Failed: ${testsFailed}`);
console.log();

if (testsFailed > 0) {
  console.log('❌ Some tests failed');
  process.exit(1);
} else {
  console.log('✅ All tests passed');
  process.exit(0);
}
