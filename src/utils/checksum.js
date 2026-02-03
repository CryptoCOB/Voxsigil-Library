/**
 * Checksum Utilities for VoxSigil Library
 * 
 * Provides SHA256 checksum computation and verification for file integrity.
 * Used by molt agents to verify agent configuration files.
 */

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

/**
 * Compute SHA256 checksum of data
 * @param {string|Buffer} data - Data to hash
 * @returns {string} Hexadecimal SHA256 hash
 */
function computeChecksum(data) {
  const hash = crypto.createHash('sha256');
  hash.update(data);
  return hash.digest('hex');
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
 * Compute checksums for all agent files
 * @param {string} agentsDir - Path to agents directory
 * @returns {Object} Map of filenames to checksums
 */
function computeAgentChecksums(agentsDir) {
  const files = ['boot.md', 'agents.md', 'memory.md', 'hooks-config.json'];
  const checksums = {};
  
  for (const file of files) {
    const filePath = path.join(agentsDir, file);
    if (fs.existsSync(filePath)) {
      checksums[file] = computeFileChecksum(filePath);
    }
  }
  
  return checksums;
}

/**
 * Verify all agent files against known checksums
 * @param {string} agentsDir - Path to agents directory
 * @param {Object} expectedChecksums - Map of filenames to expected checksums
 * @returns {Object} Verification results with status and details
 */
function verifyAgentFiles(agentsDir, expectedChecksums) {
  const results = {
    valid: true,
    files: {}
  };
  
  for (const [filename, expectedChecksum] of Object.entries(expectedChecksums)) {
    const filePath = path.join(agentsDir, filename);
    
    if (!fs.existsSync(filePath)) {
      results.valid = false;
      results.files[filename] = {
        status: 'missing',
        expected: expectedChecksum,
        actual: null
      };
      continue;
    }
    
    const actualChecksum = computeFileChecksum(filePath);
    const matches = actualChecksum === expectedChecksum;
    
    results.files[filename] = {
      status: matches ? 'valid' : 'mismatch',
      expected: expectedChecksum,
      actual: actualChecksum
    };
    
    if (!matches) {
      results.valid = false;
    }
  }
  
  return results;
}

/**
 * Generate checksum manifest for agent files
 * @param {string} agentsDir - Path to agents directory
 * @returns {Object} Manifest with checksums and metadata
 */
function generateChecksumManifest(agentsDir) {
  const checksums = computeAgentChecksums(agentsDir);
  
  return {
    version: '1.0.0',
    generated_at: new Date().toISOString(),
    algorithm: 'sha256',
    files: checksums
  };
}

/**
 * Save checksum manifest to file
 * @param {string} manifestPath - Path to save manifest
 * @param {Object} manifest - Checksum manifest object
 */
function saveChecksumManifest(manifestPath, manifest) {
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
}

/**
 * Load checksum manifest from file
 * @param {string} manifestPath - Path to manifest file
 * @returns {Object} Checksum manifest object
 */
function loadChecksumManifest(manifestPath) {
  const data = fs.readFileSync(manifestPath, 'utf8');
  return JSON.parse(data);
}

module.exports = {
  computeChecksum,
  computeFileChecksum,
  verifyFileChecksum,
  computeAgentChecksums,
  verifyAgentFiles,
  generateChecksumManifest,
  saveChecksumManifest,
  loadChecksumManifest
};
