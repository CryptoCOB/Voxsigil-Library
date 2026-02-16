import React, { useState, useEffect, useCallback, ChangeEvent } from 'react';
import modelService, {
  ChatMessage,
  ModelStatus,
  ChatRequest
} from '../utils/modelService'; // Corrected import path
import {
  CheckCircleIcon,
  XCircleIcon,
  ArrowPathIcon, // For Refresh Status button
  PaperAirplaneIcon, // For Send button
  SparklesIcon // Optional: For AI responses
} from '@heroicons/react/24/outline';
import { UserIcon } from '@heroicons/react/20/solid'; // Solid icon for user messages
import { log } from 'console'; // Removed unnecessary import

// --- (Keep other imports and types as before) ---
interface ModelInfo {
    name: string;
    source: 'ollama' | 'lmstudio'; // Make source more specific if possible
    // Add other properties if available from API
}

interface ModelApiResponse { // Define expected structure from getModels
    models: ModelInfo[];
    ollama_status: string;
    lmstudio_status: string;
}

interface StatusApiResponse { // Define expected structure from checkStatus
    ollama: { status: string };
    lmstudio: { status: string };
}


/**
 * Component for comparing and chatting with different LLM models (Ollama, LM Studio).
 */
const ModelComparison: React.FC = () => {
  // --- State ---
  const [models, setModels] = useState<ModelInfo[]>([]); // Initialize as empty array
  const [ollamaStatus, setOllamaStatus] = useState<string>('checking');
  const [lmStudioStatus, setLmStudioStatus] = useState<string>('checking');
  const [isFetchingModels, setIsFetchingModels] = useState<boolean>(true);

  const [selectedModel, setSelectedModel] = useState<string>('');
    // Ensure selectedSource type matches possible model.source values
  const [selectedSource, setSelectedSource] = useState<'ollama' | 'lmstudio' | ''>('');

  const [messageInput, setMessageInput] = useState<string>('');
  const [systemPrompt, setSystemPrompt] = useState<string>('');
  const [temperature, setTemperature] = useState<number>(0.7);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // --- Data Fetching and Status Check ---
  const fetchModelsAndStatus = useCallback(async () => {
    setIsFetchingModels(true);
    setError(null);
    console.log("Fetching models and status...");
    try {
      const [modelsResponse, statusResponse] = await Promise.all([
        modelService.getModels() as Promise<ModelApiResponse>, // Assert type if needed
        modelService.checkStatus() as Promise<StatusApiResponse> // Assert type if needed
      ]);

      // **FIX: Safely access and set models, defaulting to empty array**
      setModels(modelsResponse?.models || []); // Use || [] as fallback
      setOllamaStatus(statusResponse?.ollama?.status || 'unavailable'); // Safe access
      setLmStudioStatus(statusResponse?.lmstudio?.status || 'unavailable'); // Safe access
      console.log("Models and status fetched successfully.", modelsResponse?.models);

    } catch (err: any) {
      console.error('Error fetching models or status:', err);
      setError(`Failed to fetch models/status: ${err.message || 'Check backend and model services.'}`);
      setModels([]); // Ensure models is an empty array on error
      setOllamaStatus('unavailable');
      setLmStudioStatus('unavailable');
    } finally {
      setIsFetchingModels(false);
    }
  }, []);

  useEffect(() => {
    fetchModelsAndStatus();
  }, [fetchModelsAndStatus]);

  // --- (Keep checkStatus, handleModelSelect, sendMessage, resetChat as before, but check error conditions) ---

  const handleModelSelect = (model: ModelInfo) => {
    const isAvailable = (model.source === 'ollama' && ollamaStatus === 'available') ||
                       (model.source === 'lmstudio' && lmStudioStatus === 'available');

    if (isAvailable) {
        setSelectedModel(model.name);
        // **FIX: Ensure source type matches state type**
        setSelectedSource(model.source);
        setMessages([]);
        setError(null);
        console.log(`Selected model: ${model.name} from ${model.source}`);
    } else {
        setError(`Cannot select model: ${model.source === 'ollama' ? 'Ollama' : 'LM Studio'} service is unavailable.`);
    }
  };

  const sendMessage = async () => {
    // ... (rest of the function - Check response.message safety) ...
    setError(null); // Clear previous errors before sending
    // ... (existing checks) ...
     if (!messageInput.trim() || !selectedModel || !selectedSource) {
      setError('Please enter a message and select an available model.');
      return;
    }
    setIsSending(true);
    // ... (rest of the function - Check response.message safety) ...
    const userMessage: ChatMessage = { role: 'user', content: messageInput.trim() };
    const currentMessages = [...messages, userMessage];
    setMessages(currentMessages);
    setMessageInput('');

    try {
      const request: ChatRequest = {
        source: selectedSource,
        model: selectedModel,
        messages: currentMessages,
        temperature: temperature,
        ...(systemPrompt.trim() && { system: systemPrompt.trim() })
      };
      console.log("Sending chat request:", request);

      const response = await modelService.chat(request) as unknown as { message: ChatMessage };
      console.log("Received chat response:", response);

      // **FIX: Safely access response message**
      if (response?.message?.content) {
        setMessages(prev => [...prev, response.message]);
      } else {
         console.warn("Received response structure missing message.content", response);
         // Optionally show a generic 'empty response' message or error
         // setError('Received an empty or invalid response from the model.');
      }

    } catch (err: any) {
      console.error('Chat error:', err);
      setError(`Chat Error: ${err.response?.data?.detail || err.message || 'Failed to get response'}`);
    } finally {
      setIsSending(false);
    }
  };

   const resetChat = () => {
        setMessages([]);
        setError(null);
        setSystemPrompt('');
        console.log("Chat reset.");
    };


  // --- Filtering and Computed Values ---
  // **FIX: Add safeguard before filtering**
  const ollamaModels = (models || []).filter(model => model.source === 'ollama');
  const lmStudioModels = (models || []).filter(model => model.source === 'lmstudio');

  // Refreshes the status of Ollama and LM Studio services
  async function checkStatus(event: React.MouseEvent<HTMLButtonElement, MouseEvent>): Promise<void> {
    event.preventDefault();
    setError(null);
    setOllamaStatus('checking');
    setLmStudioStatus('checking');
    try {
      const statusResponse = await modelService.checkStatus() as StatusApiResponse;
      setOllamaStatus(statusResponse?.ollama?.status || 'unavailable');
      setLmStudioStatus(statusResponse?.lmstudio?.status || 'unavailable');
    } catch (err: any) {
      setError(`Failed to refresh status: ${err.message || 'Unknown error'}`);
      setOllamaStatus('unavailable');
      setLmStudioStatus('unavailable');
    }
  }

  // --- Render ---
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-gray-700 dark:text-gray-200">
        Model Chat Comparison
      </h1>

      {/* --- (Rest of the JSX remains the same as the previously refined version) --- */}
        {/* Model Selection Area */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Ollama */}
        <ModelSelectionCard
            title="Ollama"
            status={ollamaStatus}
            models={ollamaModels} // Pass the filtered (and safe) list
            selectedModel={selectedModel}
            selectedSource={selectedSource}
            onSelect={handleModelSelect}
            isLoading={isFetchingModels}
        />
        {/* LM Studio */}
        <ModelSelectionCard
            title="LM Studio"
            status={lmStudioStatus}
            models={lmStudioModels} // Pass the filtered (and safe) list
            selectedModel={selectedModel}
            selectedSource={selectedSource}
            onSelect={handleModelSelect}
            isLoading={isFetchingModels}
        />
      </div>

      {/* Chat Interface Card */}
      <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow">
            {/* ... (Chat Header) ... */}
             <div className="flex flex-wrap justify-between items-center mb-3 pb-2 border-b border-gray-200 dark:border-gray-700 gap-2">
                <h2 className="text-base font-semibold text-gray-700 dark:text-gray-200">
                    Chat with: {selectedModel ? `${selectedModel} (${selectedSource})` : <span className="italic text-gray-500">Select a Model</span>}
                </h2>
                <div className="flex space-x-2">
                    <button
                    className="px-2 py-1 text-xs bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md flex items-center disabled:opacity-50"
                    onClick={checkStatus}
                    disabled={ollamaStatus === 'checking' || lmStudioStatus === 'checking'}
                    aria-label="Refresh service status"
                    >
                        <ArrowPathIcon className={`w-3 h-3 mr-1 ${ollamaStatus === 'checking' || lmStudioStatus === 'checking' ? 'animate-spin' : ''}`} />
                        Status
                    </button>
                    <button
                    className="px-2 py-1 text-xs bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md"
                    onClick={resetChat}
                    >
                    Reset Chat
                    </button>
                </div>
            </div>

            {/* ... (Chat Config - System Prompt, Temp) ... */}
             <div className='grid grid-cols-1 md:grid-cols-2 gap-3 mb-3'>
                 <div> {/* System Prompt */}
                    <label htmlFor="system-prompt" className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                        System Prompt (Optional)
                    </label>
                    <textarea
                        id="system-prompt"
                        className="w-full px-2 py-1 text-xs border border-gray-300 rounded-md bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-white focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
                        rows={1}
                        placeholder="e.g., You are a helpful assistant."
                        value={systemPrompt}
                        onChange={(e) => setSystemPrompt(e.target.value)}
                    />
                </div>
                <div> {/* Temperature */}
                    <label htmlFor="temperature-range" className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Temperature: {temperature.toFixed(1)}
                    </label>
                    <input
                        id="temperature-range"
                        type="range"
                        min="0" max="2" step="0.1"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600 range-thumb-indigo"
                    />
                    <div className="flex justify-between text-[10px] text-gray-500 dark:text-gray-400 px-0.5">
                        <span>Precise</span>
                        <span>Creative</span>
                    </div>
                </div>
             </div>

            {/* ... (Messages Area) ... */}
             <div className="mb-3 border border-gray-200 dark:border-gray-700 rounded-lg overflow-y-auto h-80 bg-gray-50 dark:bg-gray-900/50">
                {messages.length > 0 ? (
                    <div className="p-3 space-y-3">
                    {messages.map((msg, index) => (
                        <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`p-2 rounded-lg max-w-[80%] ${msg.role === 'user' ? 'bg-indigo-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}>
                            <p className={`text-[10px] font-semibold mb-0.5 ${msg.role === 'user' ? 'text-indigo-100' : 'text-gray-500 dark:text-gray-400'}`}>{msg.role === 'user' ? 'You' : selectedModel}</p>
                            <p className="text-sm whitespace-pre-wrap break-words">{msg.content}</p>
                            </div>
                        </div>
                    ))}
                    {isSending && ( /* Loading Indicator */
                        <div className="flex justify-start">
                           <div className="p-2 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 inline-flex items-center space-x-1">
                               <span className="text-sm italic text-gray-500 dark:text-gray-400">typing</span>
                               <div className="h-1.5 w-1.5 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s'}}></div>
                               <div className="h-1.5 w-1.5 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s'}}></div>
                               <div className="h-1.5 w-1.5 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.3s'}}></div>
                           </div>
                       </div>
                    )}
                    </div>
                ) : ( /* Empty State */
                    <div className="flex items-center justify-center h-full text-center text-gray-500 dark:text-gray-400 px-4">
                       <p className="text-sm">{selectedModel ? 'Start the conversation by typing below.' : 'Select an available model above to begin chatting.'}</p>
                    </div>
                )}
             </div>

            {/* ... (Error Display) ... */}
            {error && (
                <div className="mb-3 p-2 bg-red-100 dark:bg-red-900/30 border border-red-300 dark:border-red-700 text-red-700 dark:text-red-300 text-xs rounded-lg flex items-center gap-2">
                    <XCircleIcon className="h-4 w-4 flex-shrink-0" aria-hidden="true" />
                    {error}
                </div>
            )}

            {/* ... (Input Area) ... */}
             <div className="flex items-end gap-2">
                <textarea
                    className="flex-grow px-2 py-1.5 text-sm border border-gray-300 rounded-md bg-white dark:bg-gray-700 dark:border-gray-600 dark:text-white focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 resize-none"
                    placeholder={selectedModel ? `Message ${selectedModel}...` : 'Select a model first...'}
                    rows={1}
                    value={messageInput}
                    onChange={(e) => setMessageInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!isSending) sendMessage();} }}
                    disabled={!selectedModel || isSending}
                    aria-label="Chat message input"
                />
                <button
                    className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded-md flex items-center justify-center disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed"
                    onClick={sendMessage}
                    disabled={isSending || !messageInput.trim() || !selectedModel}
                    aria-label="Send message"
                >
                    <PaperAirplaneIcon className={`w-4 h-4 ${isSending ? 'hidden' : 'inline'}`} />
                    <span className={` ${isSending ? 'inline' : 'hidden'} w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin`}></span>
                </button>
            </div>
      </div>
    </div>
  );
};

// --- (Keep ModelSelectionCard Component as before) ---
interface ModelSelectionCardProps {
    title: string;
    status: string;
    models: ModelInfo[];
    selectedModel: string;
    selectedSource: string; // Type needs to match state: 'ollama' | 'lmstudio' | ''
    onSelect: (model: ModelInfo) => void;
    isLoading: boolean;
}

const ModelSelectionCard: React.FC<ModelSelectionCardProps> = ({ title, status, models, selectedModel, selectedSource, onSelect, isLoading }) => {
  const sourceName = title.toLowerCase().replace(' ', '') as 'ollama' | 'lmstudio'; // Ensure type consistency
  const isServiceAvailable = status === 'available';

  return (
    <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-base font-semibold text-gray-700 dark:text-gray-200">{title}</h2>
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
            status === 'available' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
            status === 'checking' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200 animate-pulse' :
            'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
          }`}
        >
          {status === 'available' ? <CheckCircleIcon className="h-3 w-3 mr-1" /> :
           status === 'checking' ? <ArrowPathIcon className="h-3 w-3 mr-1 animate-spin" /> :
           <XCircleIcon className="h-3 w-3 mr-1" />}
           {status.charAt(0).toUpperCase() + status.slice(1)}
        </span>
      </div>

      {isLoading ? (
          <div className="h-16 flex items-center justify-center text-xs text-gray-500">Loading models...</div>
      ) : models && models.length > 0 ? ( // Check if models is truthy before checking length
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 mt-1 max-h-48 overflow-y-auto pr-1">
          {models.map((model) => (
            <button
              key={`${sourceName}-${model.name}`}
              disabled={!isServiceAvailable}
              className={`w-full text-left px-2 py-1 rounded text-[11px] transition-colors truncate ${
                selectedModel === model.name && selectedSource === sourceName // Compare with specific sourceName
                  ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-200 font-semibold ring-1 ring-indigo-300 dark:ring-indigo-700'
                  : isServiceAvailable
                    ? 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 focus:outline-none focus:ring-1 focus:ring-indigo-400'
                    : 'bg-gray-100 text-gray-400 dark:bg-gray-700 dark:text-gray-500 cursor-not-allowed'
              }`}
              onClick={() => onSelect(model)}
              title={model.name}
            >
              {model.name}
            </button>
          ))}
        </div>
      ) : (
        <p className="text-xs text-gray-500 dark:text-gray-400 italic mt-2">
          No models found. Ensure {title} service is running and accessible.
        </p>
      )}
    </div>
  );
};


export default ModelComparison;