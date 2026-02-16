/**
 * CSS Import Rules Fixer
 * 
 * This module helps prevent the "@import rules are not allowed here" warning
 * that occurs in dynamically imported CSS in Vite/React applications.
 * 
 * The issue comes from constructable stylesheets trying to process @import
 * rules in an inappropriate context.
 */

/**
 * Patch the CSSStyleSheet prototype to handle @import rules properly
 */
export function fixCSSImportRules() {
  // Only run in browser environment
  if (typeof window === 'undefined') return;

  try {
    // Keep track of the original replaceSync method
    const originalReplaceSync = CSSStyleSheet.prototype.replaceSync;

    // Override the replaceSync method
    CSSStyleSheet.prototype.replaceSync = function(cssText) {
      // Process the CSS text to handle @import rules
      const processedCssText = processImportRules(cssText);
      
      // Call the original method with the processed CSS
      return originalReplaceSync.call(this, processedCssText);
    };

    // Silently log success
    console.debug('CSS Import Rules Fixer: Successfully patched CSSStyleSheet.prototype.replaceSync');
  } catch (error) {
    // Don't break the application if patching fails
    console.debug('CSS Import Rules Fixer: Failed to patch CSSStyleSheet', error);
  }
}

/**
 * Process CSS text to handle @import rules
 * @param {string} cssText - The CSS text to process
 * @returns {string} - The processed CSS text
 */
function processImportRules(cssText) {
  if (!cssText || typeof cssText !== 'string') return cssText;

  try {
    // Simple regex to find @import rules
    const importRegex = /@import\s+(['"])([^'"]+)\1;?/g;
    
    // Replace @import rules with comments
    const processedText = cssText.replace(importRegex, 
      (match) => `/* Removed import rule: ${match} */`);

    return processedText;
  } catch (error) {
    return cssText; // Return original if processing fails
  }
}

// Auto-initialize when imported
fixCSSImportRules();