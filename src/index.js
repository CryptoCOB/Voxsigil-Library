/**
 * VoxSigil Library - Molt Agent Integration SDK
 * 
 * Main entry point for JavaScript/Node.js environments.
 * Provides utilities for molt agents to interact with VoxSigil prediction markets.
 * 
 * @module @voxsigil/library
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

/**
 * Load agent configuration from the agents directory
 * @returns {Object} Agent configuration with boot, agents, and memory templates
 */
function loadAgentConfig() {
  const agentsDir = path.join(__dirname, 'agents');
  
  return {
    boot: fs.readFileSync(path.join(agentsDir, 'boot.md'), 'utf8'),
    agents: fs.readFileSync(path.join(agentsDir, 'agents.md'), 'utf8'),
    memory: fs.readFileSync(path.join(agentsDir, 'memory.md'), 'utf8'),
    hooks: JSON.parse(fs.readFileSync(path.join(agentsDir, 'hooks-config.json'), 'utf8'))
  };
}

/**
 * Compute SHA256 checksum of a file or string
 * @param {string|Buffer} data - Data to hash
 * @returns {string} Hexadecimal SHA256 hash
 */
function computeChecksum(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

/**
 * Compute SHA256 checksum of a file
 * @param {string} filePath - Path to file
 * @returns {string} Hexadecimal SHA256 hash
 */
function computeFileChecksum(filePath) {
  const data = fs.readFileSync(filePath);
  return computeChecksum(data);
}

/**
 * Verify file integrity using SHA256 checksum
 * @param {string} filePath - Path to file
 * @param {string} expectedChecksum - Expected SHA256 hash
 * @returns {boolean} True if checksum matches
 */
function verifyFileChecksum(filePath, expectedChecksum) {
  const actualChecksum = computeFileChecksum(filePath);
  return actualChecksum === expectedChecksum;
}

/**
 * Get VoxSigil agent metadata
 * @returns {Object} Metadata about the library and agent capabilities
 */
function getMetadata() {
  const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, '../package.json'), 'utf8'));
  
  return {
    name: pkg.name,
    version: pkg.version,
    description: pkg.description,
    repository: pkg.repository.url,
    keywords: pkg.keywords,
    capabilities: [
      'market-analysis',
      'signal-broadcasting',
      'agent-coordination',
      'prediction-markets'
    ],
    endpoints: {
      github: 'https://github.com/CryptoCOB/Voxsigil-Library',
      docs: 'https://voxsigil.online/docs',
      api: 'https://voxsigil.online/api'
    }
  };
}

module.exports = {
  loadAgentConfig,
  computeChecksum,
  computeFileChecksum,
  verifyFileChecksum,
  getMetadata
};
