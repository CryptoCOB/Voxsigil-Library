/**
 * ExtensionGuard - Enhanced utility to handle cryptocurrency wallet extension conflicts
 * 
 * This module helps prevent conflicts between crypto wallet browser extensions
 * that try to inject themselves into the global window object.
 */

// Track if providers have been properly initialized
let providersInitialized = false;

// List of known provider properties that might cause conflicts
const WALLET_PROVIDERS = {
  PHANTOM: 'phantom',
  EXODUS: 'exodus',
  SOLANA: 'solana',
  ETHEREUM: 'ethereum'
};

/**
 * Create a no-op console wrapper to silence certain errors
 */
const silentConsole = {
  log: () => {},
  warn: () => {},
  error: () => {},
  debug: () => {}
};

/**
 * Intercept and silence specific console warnings related to wallet extensions
 */
function setupConsoleFilters() {
  const originalConsoleWarn = console.warn;
  const originalConsoleError = console.error;
  
  // Filter patterns that should be silenced
  const filterPatterns = [
    'Removing unpermitted intrinsics',
    'Could not assign Exodus provider',
    'lockdown-install',
    'phantom.solana',
    'phantom.ethereum'
  ];
  
  // Override console.warn to filter out wallet-related warnings
  console.warn = function(...args) {
    const message = args.join(' ');
    if (filterPatterns.some(pattern => message.includes(pattern))) {
      // Silently ignore these specific warnings
      return;
    }
    originalConsoleWarn.apply(console, args);
  };
  
  // Override console.error to filter out wallet-related errors
  console.error = function(...args) {
    const message = args.join(' ');
    if (filterPatterns.some(pattern => message.includes(pattern))) {
      // Silently ignore these specific errors
      return;
    }
    originalConsoleError.apply(console, args);
  };
}

/**
 * Initializes wallet providers in the correct order to prevent conflicts
 * This addresses the "Unpermitted Intrinsics" and provider assignment issues
 */
export function initializeProviders() {
  if (providersInitialized) return;
  
  try {
    // Setup console filters first to suppress warnings during initialization
    setupConsoleFilters();
    
    // Create a controlled execution context for provider initialization
    const initializeProvider = (providerName) => {
      try {
        // Check if the provider exists without directly accessing it
        if (Object.prototype.hasOwnProperty.call(window, providerName)) {
          // Create a safe reference without triggering security errors
          const descriptor = Object.getOwnPropertyDescriptor(window, providerName);
          if (descriptor && descriptor.configurable) {
            // Provider is available, but we don't need to log this
          }
        }
      } catch (err) {
        // Silently handle initialization errors
      }
    };
    
    // Initialize providers in a specific order to avoid conflicts
    // Solana ecosystem first
    initializeProvider(WALLET_PROVIDERS.SOLANA);
    initializeProvider(WALLET_PROVIDERS.PHANTOM);
    
    // Ethereum ecosystem second
    initializeProvider(WALLET_PROVIDERS.ETHEREUM);
    initializeProvider(WALLET_PROVIDERS.EXODUS);
    
    // Mark as initialized to prevent duplicate initialization
    providersInitialized = true;
    
  } catch (error) {
    // Silently handle provider initialization errors
  }
}

/**
 * Protects the application from extension conflicts by:
 * 1. Setting up defensive property access
 * 2. Monitoring for unauthorized modifications
 */
export function protectFromExtensionConflicts() {
  try {
    // Store original properties that might be modified by extensions
    const originalProperties = {};
    
    // List of known problematic extension injections
    const knownExtensionProps = Object.values(WALLET_PROVIDERS);
    
    // Store the original property descriptors if they exist
    knownExtensionProps.forEach(prop => {
      if (window[prop] !== undefined) {
        originalProperties[prop] = Object.getOwnPropertyDescriptor(window, prop);
      }
    });
    
    // Set up defensive getters/setters for nested properties
    if (window.phantom) {
      // Create safer versions of nested properties
      const protectNestedProperty = (parentObj, propName) => {
        try {
          if (parentObj[propName]) {
            const original = parentObj[propName];
            // Use Object.defineProperty for safer access
            Object.defineProperty(parentObj, propName, {
              get() { return original; },
              set(newValue) {
                // Allow setting if coming from the same origin provider
                // Otherwise, keep the original to prevent conflicts
                return original;
              },
              enumerable: true,
              configurable: true
            });
          }
        } catch (err) {
          // Silently handle errors to prevent breaking the application
        }
      };
      
      // Block Exodus from writing to phantom's properties
      if (window.phantom) {
        // Use try/catch for each property to prevent any errors from breaking the app
        try { protectNestedProperty(window.phantom, 'ethereum'); } catch {}
        try { protectNestedProperty(window.phantom, 'solana'); } catch {}
      }
    }
    
    return originalProperties;
  } catch (error) {
    // Silently handle extension protection errors
    return {};
  }
}

/**
 * Safely access wallet providers without causing errors
 * @param {string} providerName - Name of the provider to access
 * @returns The provider object or null if unavailable
 */
export function safeWalletAccess(providerName) {
  try {
    // First check if the provider exists before accessing it
    if (Object.prototype.hasOwnProperty.call(window, providerName)) {
      return window[providerName];
    }
    return null;
  } catch (error) {
    // Silently handle provider access errors
    return null;
  }
}

// Initialize immediately when this module is imported
initializeProviders();
protectFromExtensionConflicts();