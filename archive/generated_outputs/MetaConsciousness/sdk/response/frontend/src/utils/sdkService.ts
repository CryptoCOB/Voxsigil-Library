/**
 * SDK Service
 * 
 * Service for interacting with the MetaConsciousness SDK through the backend API
 */

import { API_BASE_URL } from './constants';

/**
 * Interface for a query request to the SDK
 */
interface SDKQueryRequest {
  query: string;
  params?: Record<string, any>;
}

/**
 * Interface for a belief analysis request
 */
interface BeliefAnalysisRequest {
  text: string;
  context?: Record<string, any>;
}

/**
 * Interface for SDK component information
 */
interface SDKComponent {
  name: string;
  type: string;
  methods: string[];
}

/**
 * Service class for interacting with the MetaConsciousness SDK
 */
class SDKService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = `${API_BASE_URL}/api/sdk`;
  }

  /**
   * Get the current status of the SDK integration
   */
  async getStatus(): Promise<Record<string, any>> {
    try {
      const response = await fetch(`${this.baseUrl}/status`);
      if (!response.ok) {
        throw new Error(`Failed to get SDK status: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error getting SDK status:', error);
      throw error;
    }
  }

  /**
   * Get a list of available SDK components
   */
  async getComponents(): Promise<Record<string, SDKComponent>> {
    try {
      const response = await fetch(`${this.baseUrl}/components`);
      if (!response.ok) {
        throw new Error(`Failed to get SDK components: ${response.statusText}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error getting SDK components:', error);
      throw error;
    }
  }

  /**
   * Process a query using the MetaConsciousness SDK
   * 
   * @param query The query text to process
   * @param params Additional parameters for processing
   */
  async processQuery(query: string, params?: Record<string, any>): Promise<Record<string, any>> {
    try {
      const request: SDKQueryRequest = { query, params };
      const response = await fetch(`${this.baseUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`Failed to process query: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error processing query with SDK:', error);
      throw error;
    }
  }

  /**
   * Analyze text for beliefs using the SDK
   * 
   * @param text The text to analyze
   * @param context Optional context information
   */
  async analyzeBeliefs(text: string, context?: Record<string, any>): Promise<Record<string, any>> {
    try {
      const request: BeliefAnalysisRequest = { text, context };
      const response = await fetch(`${this.baseUrl}/belief-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`Failed to analyze beliefs: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error analyzing beliefs with SDK:', error);
      throw error;
    }
  }
}

// Create and export a singleton instance
const sdkService = new SDKService();
export default sdkService;