// import axios from 'axios'; // Removed unused import

// API base URL - adjust this to match your backend API endpoint
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

// Model Service for integrating with Ollama and LM Studio APIs

// Type definitions
export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ModelInfo {
  name: string;
  source: 'ollama' | 'lmstudio';
  details?: any; // Additional model-specific information
}

export interface ChatRequest {
  source: 'ollama' | 'lmstudio';
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  system?: string;
}

export interface ChatResponse {
  source: 'ollama' | 'lmstudio';
  model: string;
  message: ChatMessage;
  raw_response: any;
}

export interface GenerateRequest {
  source: 'ollama' | 'lmstudio';
  model: string;
  prompt: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  system?: string;
}

export interface GenerateResponse {
  source: 'ollama' | 'lmstudio';
  model: string;
  text: string;
  raw_response: any;
}

export interface ModelStatus {
  ollama: {
    status: 'available' | 'unavailable' | 'error';
    message: string;
  };
  lmstudio: {
    status: 'available' | 'unavailable' | 'error';
    message: string;
  };
}

// Default API URLs - these can be updated from settings
// const MODEL_API_PATH = '/api/models'; // Removed unused variable

// Service class for interacting with model APIs
export class ModelService {
  chat(request: ChatRequest) {
    throw new Error('Method not implemented.');
  }
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Content-Type': 'application/json',
    };
  }

  // Method to update the base URL (e.g., when settings change)
  setBaseUrl(url: string) {
    this.baseUrl = url;
  }

  // Get all available models from both Ollama and LM Studio
  async getModels(): Promise<{ models: ModelInfo[], ollama_status: string, lmstudio_status: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/models`, {
        method: 'GET',
        headers: this.headers,
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error fetching models:', error);
      return { 
        models: [], 
        ollama_status: 'error', 
        lmstudio_status: 'error' 
      };
    }
  }

  async checkStatus() {
    // Example implementation: Replace URLs and logic as needed
    const ollamaUrl = localStorage.getItem('ollamaUrl') || 'http://localhost:11434';
    const lmstudioUrl = localStorage.getItem('lmStudioUrl') || 'http://localhost:1234';

    const check = async (url: string) => {
      try {
        const res = await fetch(url, { method: 'GET' });
        if (res.ok) return { status: 'available' };
        return { status: 'unavailable' };
      } catch {
        return { status: 'unavailable' };
      }
    };

    const [ollama, lmstudio] = await Promise.all([
      check(ollamaUrl),
      check(lmstudioUrl),
    ]);

    return { ollama, lmstudio };
  }

  // Check the status of model services by calling backend endpoints
  async checkModelStatus(): Promise<ModelStatus> {
    let ollamaResult: { status: 'available' | 'unavailable' | 'error'; message: string } = { status: 'checking', message: 'Checking...' };
    let lmstudioResult: { status: 'available' | 'unavailable' | 'error'; message: string } = { status: 'checking', message: 'Checking...' };

    try {
      // Check Ollama status via backend
      const ollamaResponse = await fetch(`${this.baseUrl}/models/ollama/status`, {
        method: 'GET',
        headers: this.headers,
      });
      if (!ollamaResponse.ok) {
        // Try to parse error message from backend if available
        let errorMsg = `Ollama status check failed: ${ollamaResponse.status}`;
        try {
            const errorData = await ollamaResponse.json();
            errorMsg = errorData.message || errorMsg;
        } catch {}
        throw new Error(errorMsg);
      }
      ollamaResult = await ollamaResponse.json();

    } catch (error) {
      console.error('Error checking Ollama status via backend:', error);
      ollamaResult = { status: 'error', message: error.message || 'Failed to connect to backend for Ollama status' };
    }

    try {
      // Check LM Studio status via backend
      const lmstudioResponse = await fetch(`${this.baseUrl}/models/lmstudio/status`, {
        method: 'GET',
        headers: this.headers,
      });
      if (!lmstudioResponse.ok) {
        // Try to parse error message from backend if available
        let errorMsg = `LM Studio status check failed: ${lmstudioResponse.status}`;
        try {
            const errorData = await lmstudioResponse.json();
            errorMsg = errorData.message || errorMsg;
        } catch {}
        throw new Error(errorMsg);
      }
      lmstudioResult = await lmstudioResponse.json();

    } catch (error) {
      console.error('Error checking LM Studio status via backend:', error);
      lmstudioResult = { status: 'error', message: error.message || 'Failed to connect to backend for LM Studio status' };
    }

    return {
      ollama: ollamaResult,
      lmstudio: lmstudioResult
    };
  }

  // Send a chat completion request
  async chatCompletion(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/models/chat`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to get chat completion: ${response.status} - ${errorText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error getting chat completion:', error);
      throw new Error(`Chat completion failed: ${error.message}`);
    }
  }

  // Send a text generation request
  async generateText(request: GenerateRequest): Promise<GenerateResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/models/generate`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to generate text: ${response.status} - ${errorText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error generating text:', error);
      throw new Error(`Text generation failed: ${error.message}`);
    }
  }

  // Helper method to save configuration settings
  async saveSettings(settings: Record<string, any>): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/settings`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(settings),
      });
      
      return response.ok;
    } catch (error) {
      console.error('Error saving settings:', error);
      return false;
    }
  }
}


// Removed invalid standalone async checkStatus function


// Create a singleton instance for use throughout the app
export const modelService = new ModelService();

export default modelService;