/// <reference types="vite/client" />
import React, { useState, useEffect, useCallback, ChangeEvent } from 'react';
import axios from 'axios'; // Using axios for API calls
import {
  MagnifyingGlassIcon,
  TagIcon,
  XMarkIcon,
  PlusIcon,
  ChatBubbleLeftRightIcon // Icon for placeholder
} from '@heroicons/react/24/outline';
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/20/solid'; // Solid icons for pagination

// --- Configuration ---
// TODO: Move API URL to config/env variable
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const ITEMS_PER_PAGE = 10;
const SEARCH_DEBOUNCE_MS = 500;

// --- Custom Hook for Debouncing ---
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    // Cancel the timeout if value changes (also on delay change or unmount)
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}


// --- Type Definitions (align with Backend LLMEntry.to_dict) ---
interface TokenData {
    prompt_tokens: number | null;
    completion_tokens: number | null;
    total_tokens: number | null;
}

interface Metadata {
    latency_seconds: number | null;
    tokens: TokenData | null;
    // Add cognitive_signals structure here if needed later
}
type Message = {
  role: string;
  content: string;
};

interface InteractionEntry {
  id: string; // UUIDs are strings
  query: string | Message[]; // Adjust based on actual prompt type stored/returned
  response: string | null; // Allow null response
  model: string;
  source: string;
  type: 'text' | 'image' | 'embed';
  tags: string[]; // Added tags
  created_at: string; // ISO string timestamp
  metadata: Metadata;
}

// Interface for API response format
interface HistoryResponse {
  results: InteractionEntry[];
  total_found: number;
}

// --- Helper Components (Optional but recommended for larger components) ---

// Example: Pagination Controls Component
interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

const PaginationControls: React.FC<PaginationProps> = ({ currentPage, totalPages, onPageChange }) => {
  if (totalPages <= 1) return null;

  const handlePrevious = () => onPageChange(Math.max(1, currentPage - 1));
  const handleNext = () => onPageChange(Math.min(totalPages, currentPage + 1));

  return (
    <div className="px-3 py-2 flex items-center justify-between border-t border-gray-200 dark:border-gray-700 sm:px-4">
        <button
          onClick={handlePrevious}
          disabled={currentPage === 1}
          className="inline-flex items-center px-3 py-1 border border-gray-300 dark:border-gray-600 text-xs font-medium rounded-md text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronLeftIcon className="h-4 w-4 mr-1" aria-hidden="true" />
          Previous
        </button>
        <span className="text-xs text-gray-600 dark:text-gray-400">
          Page {currentPage} of {totalPages}
        </span>
        <button
          onClick={handleNext}
          disabled={currentPage === totalPages}
          className="inline-flex items-center px-3 py-1 border border-gray-300 dark:border-gray-600 text-xs font-medium rounded-md text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Next
          <ChevronRightIcon className="h-4 w-4 ml-1" aria-hidden="true" />
        </button>
    </div>
  );
};


// --- Main History Component ---

const History: React.FC = () => {
  // --- State ---
  const [interactions, setInteractions] = useState<InteractionEntry[]>([]);
  const [selectedInteraction, setSelectedInteraction] = useState<InteractionEntry | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filtering, Searching, Pagination State
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [filterModel, setFilterModel] = useState<string>('all'); // 'all', 'GPT-5', 'Claude-3', etc.
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalItems, setTotalItems] = useState<number>(0);

  // Tag Editing State
  const [tagInput, setTagInput] = useState<string>('');
  const [isTagLoading, setIsTagLoading] = useState<boolean>(false);

  // Debounced Search Term
  const debouncedSearchTerm = useDebounce(searchTerm, SEARCH_DEBOUNCE_MS);

  const totalPages = Math.ceil(totalItems / ITEMS_PER_PAGE);

  // --- Data Fetching ---
  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    const offset = (currentPage - 1) * ITEMS_PER_PAGE;
    const limit = ITEMS_PER_PAGE;

    let url = `${API_BASE_URL}/api/`;
    const params: Record<string, any> = { limit, offset };

    if (debouncedSearchTerm.trim()) {
        // Use search endpoint if searching
        url += 'search';
        params.term = debouncedSearchTerm.trim();
    } else {
        // Use history endpoint otherwise
        url += 'history';
        if (filterModel !== 'all') {
            params.model = filterModel; // Add model filter if not 'all'
        }
    }

    try {
      console.log(`Fetching data from ${url} with params:`, params); // Debug log
      const response = await axios.get<HistoryResponse>(url, { params });

      // Updated to handle the new response structure
      if (response.data && response.data.results) {
          const fetchedInteractions = response.data.results;
          const fetchedTotal = response.data.total_found;
          
          setInteractions(fetchedInteractions);
          setTotalItems(fetchedTotal);
          
          // If we had a selected interaction but it's no longer in the results, clear it
          if (selectedInteraction && !fetchedInteractions.find(i => i.id === selectedInteraction.id)) {
              setSelectedInteraction(null);
          }
      } else {
          console.error("Unexpected API response format:", response.data);
          throw new Error("Unexpected API response format received.");
      }

    } catch (err: any) {
      console.error("Failed to fetch interaction history:", err);
      setError(err.response?.data?.detail || err.message || 'Failed to fetch data. Please try again.');
      setInteractions([]); // Clear data on error
      setTotalItems(0);
    } finally {
      setIsLoading(false);
    }
  }, [currentPage, filterModel, debouncedSearchTerm]);

  // Effect to fetch data when filters, search term (debounced), or page changes
  useEffect(() => {
    fetchData();
  }, [fetchData]); // fetchData includes all dependencies

  // Reset page to 1 when filter or search term changes
  useEffect(() => {
    setCurrentPage(1);
  }, [filterModel, debouncedSearchTerm]);

  // --- Event Handlers ---
  const handleFilterChange = (newFilter: string) => {
    setFilterModel(newFilter);
    setSelectedInteraction(null); // Clear selection on filter change
  };

  const handleSearchChange = (event: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
    setSelectedInteraction(null); // Clear selection on search change
  };

  const handlePageChange = (pageNumber: number) => {
    setCurrentPage(pageNumber);
    setSelectedInteraction(null); // Clear selection on page change
  };

  const handleInteractionClick = (interaction: InteractionEntry) => {
    setSelectedInteraction(interaction);
  };

  // --- Tag Management Handlers ---
  const handleAddTag = async () => {
    if (!selectedInteraction || !tagInput.trim()) return;
    setIsTagLoading(true);
    setError(null);
    const tagToAdd = tagInput.trim();

    try {
      const response = await axios.post<InteractionEntry>(
        `${API_BASE_URL}/api/history/${selectedInteraction.id}/tags`,
        { tags: [tagToAdd] } // API expects a list of tags
      );
      // Update the selected interaction and the list with the new data from response
      const updatedInteraction = response.data;
      setSelectedInteraction(updatedInteraction);
      setInteractions(prev =>
        prev.map(item => item.id === updatedInteraction.id ? updatedInteraction : item)
      );
      setTagInput(''); // Clear input
      console.info(`Successfully added tag "${tagToAdd}"`);
    } catch (err: any) {
        console.error("Failed to add tag:", err);
        setError(err.response?.data?.detail || err.message || 'Failed to add tag.');
    } finally {
        setIsTagLoading(false);
    }
  };

  const handleRemoveTag = async (tagToRemove: string) => {
    if (!selectedInteraction) return;
    setIsTagLoading(true);
    setError(null);

     try {
      // Note: Axios DELETE typically sends data in `config.data`, not directly as 2nd arg
      const response = await axios.delete<InteractionEntry>(
        `${API_BASE_URL}/api/history/${selectedInteraction.id}/tags`,
        { data: { tags: [tagToRemove] } } // Send tags to remove in request body
      );
      const updatedInteraction = response.data;
      setSelectedInteraction(updatedInteraction);
      setInteractions(prev =>
        prev.map(item => item.id === updatedInteraction.id ? updatedInteraction : item)
      );
      console.info(`Successfully removed tag "${tagToRemove}"`);
    } catch (err: any) {
        console.error("Failed to remove tag:", err);
        setError(err.response?.data?.detail || err.message || 'Failed to remove tag.');
    } finally {
        setIsTagLoading(false);
    }
  };
  
  // Helper to display query (string or messages)
  const renderPrompt = (query: InteractionEntry['query']): string => {
    if (typeof query === 'string') {
      return query;
    } else if (Array.isArray(query)) {
      // Simple join for display, could format better (e.g., Role: Content)
      return query.map((msg: Message) => msg?.content || '').join('\n');
    }
    return 'N/A';
  };

  // --- Render ---
  return (
    // Padding provided by App.tsx layout
    <div className="max-w-full">
      {/* Consistent Heading */}
      <h1 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3">
        Conversation History
      </h1>

      {/* Filters and Search */}
      <div className="mb-3 p-2 bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="flex flex-col sm:flex-row justify-between items-center gap-2">
          {/* Filter Buttons */}
          <div className="flex flex-wrap gap-1">
             {['all', 'GPT-5', 'Claude-3', 'Gemini-3'].map((modelFilter) => ( // Example models
               <button
                  key={modelFilter}
                  onClick={() => handleFilterChange(modelFilter)}
                  className={`px-2 py-0.5 text-xs font-medium rounded-md transition-colors ${
                     filterModel === modelFilter
                       ? 'bg-indigo-600 text-white'
                       : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600'
                  }`}
               >
                  {modelFilter === 'all' ? 'All Models' : modelFilter}
               </button>
             ))}
          </div>

          {/* Search Input */}
          <div className="relative w-full sm:w-64">
            <div className="absolute inset-y-0 left-0 pl-2 flex items-center pointer-events-none">
              <MagnifyingGlassIcon className="h-3 w-3 text-gray-400" aria-hidden="true" />
            </div>
            <input
              type="search"
              placeholder="Search query/response..."
              value={searchTerm}
              onChange={handleSearchChange}
              className="block w-full pl-7 pr-2 py-1 border border-gray-300 rounded-md text-xs
                         leading-5 bg-white dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300
                         placeholder-gray-500 dark:placeholder-gray-400
                         focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
        </div>
        {/* Display API Error */}
        {error && <p className="mt-1 text-xs text-red-600 dark:text-red-400">{error}</p>}
      </div>

      {/* Conversations list and detail view */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">

        {/* Interaction list */}
        <div className="lg:col-span-1 flex flex-col">
          <div className="bg-white dark:bg-gray-800 shadow rounded-lg overflow-hidden flex-grow flex flex-col">
             {/* Header with count */}
             <div className="px-2 py-1 text-xs font-medium text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
                {totalItems} Interaction{totalItems !== 1 ? 's' : ''}
             </div>
             <div className="flex-grow overflow-y-auto">
                {isLoading ? (
                <div className="p-4 flex justify-center items-center h-full">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-500"></div>
                </div>
                ) : (
                <ul className="divide-y divide-gray-200 dark:divide-gray-700">
                    {interactions.length > 0 ? (
                    interactions.map((interaction) => (
                        <li
                        key={interaction.id}
                        onClick={() => handleInteractionClick(interaction)}
                        className={`block px-2 py-1.5 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors ${
                            selectedInteraction?.id === interaction.id ? 'bg-indigo-50 dark:bg-indigo-900/30' : ''
                        }`}
                        >
                            <div className="flex items-center justify-between text-[10px] mb-0.5">
                                <span className="font-medium text-gray-600 dark:text-gray-300">{interaction.model}</span>
                                <span className="text-gray-500 dark:text-gray-400">
                                    {new Date(interaction.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                                </span>
                            </div>
                            <p className="text-xs font-medium text-gray-900 dark:text-gray-100 truncate" title={renderPrompt(interaction.query)}>
                                {renderPrompt(interaction.query)}
                            </p>
                            {/* Display Tags - Make even smaller & more compact */}
                            {interaction.tags && interaction.tags.length > 0 && (
                                <div className="mt-1 flex flex-wrap gap-0.5">
                                {interaction.tags.map(tag => (
                                    <span key={tag} className="inline-flex items-center px-1 py-0 rounded text-[8px] font-medium bg-gray-100 dark:bg-gray-700/80 text-gray-700 dark:text-gray-300">
                                      <TagIcon className="h-2 w-2 mr-0.5" /> {tag}
                                    </span>
                                ))}
                                </div>
                            )}
                        </li>
                    ))
                    ) : (
                    <li className="p-4 text-center text-xs text-gray-500 dark:text-gray-400">
                        No conversation history found matching your criteria.
                    </li>
                    )}
                </ul>
                )}
            </div>

            {/* Pagination */}
            <PaginationControls
                currentPage={currentPage}
                totalPages={totalPages}
                onPageChange={handlePageChange}
            />
          </div>
        </div>

        {/* Conversation detail */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 shadow rounded-lg h-[calc(100vh-10rem)]"> {/* Adjust height as needed */}
            {selectedInteraction ? (
              <div className="p-2 h-full flex flex-col">
                {/* Detail Header - More compact */}
                <div className="flex justify-between items-start mb-2 pb-1 border-b border-gray-200 dark:border-gray-700">
                  <div>
                     <h3 className="text-sm font-semibold text-gray-900 dark:text-white leading-tight">
                        Interaction Detail
                     </h3>
                      <p className="text-[9px] text-gray-500 dark:text-gray-400">ID: {selectedInteraction.id}</p>
                  </div>
                  <div className="text-right flex-shrink-0 ml-2">
                     <span className="inline-flex items-center px-1.5 py-0 rounded text-[10px] font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-800 dark:text-indigo-100">
                      {selectedInteraction.model} ({selectedInteraction.source})
                    </span>
                     <p className="text-[9px] text-gray-500 dark:text-gray-400 mt-0.5">
                      {new Date(selectedInteraction.created_at).toLocaleString()}
                     </p>
                  </div>
                </div>

                 {/* Scrollable Content */}
                <div className="flex-grow overflow-y-auto pr-1 space-y-2 text-xs">
                  {/* Content Grid - Better organization */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-2 mb-2">
                    {/* Query Section */}
                    <div className="lg:col-span-2">
                      <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-0.5 text-[10px] uppercase tracking-wider">Query</h4>
                      <div className="bg-indigo-50 dark:bg-gray-700/30 p-1.5 rounded">
                        <p className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap break-words text-[11px]">{renderPrompt(selectedInteraction.query)}</p>
                      </div>
                    </div>

                    {/* Response Section */}
                    <div className="lg:col-span-2">
                      <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-0.5 text-[10px] uppercase tracking-wider">Response</h4>
                      <div className="bg-gray-50 dark:bg-gray-700/50 p-1.5 rounded">
                        <p className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap break-words text-[11px]">
                          {selectedInteraction.response || <span className="italic text-gray-400">No response content</span>}
                        </p>
                      </div>
                    </div>

                    {/* Metadata Section - 2 columns side by side */}
                    <div>
                      <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-0.5 text-[10px] uppercase tracking-wider">Metadata</h4>
                      <div className="grid grid-cols-2 gap-1 text-[10px]">
                        <div className="bg-gray-50 dark:bg-gray-700/20 p-1.5 rounded">
                          <p className="text-gray-500 dark:text-gray-400">Latency</p>
                          <p className="font-medium text-gray-800 dark:text-gray-200">
                            {selectedInteraction.metadata?.latency_seconds?.toFixed(2) ?? 'N/A'}s
                          </p>
                        </div>
                        <div className="bg-gray-50 dark:bg-gray-700/20 p-1.5 rounded">
                          <p className="text-gray-500 dark:text-gray-400">Tokens</p>
                          <p className="font-medium text-gray-800 dark:text-gray-200">
                            {selectedInteraction.metadata?.tokens?.total_tokens ?? 'N/A'}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Tagging Section */}
                    <div>
                      <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-0.5 text-[10px] uppercase tracking-wider">Tags</h4>
                      <div className="bg-gray-50 dark:bg-gray-700/20 p-1.5 rounded h-full">
                        {/* Tags Section - Compact grid layout */}
                        <div className="flex flex-wrap gap-1 mb-1">
                          {selectedInteraction.tags.length > 0 ? selectedInteraction.tags.map(tag => (
                            <span key={tag} className="relative group inline-flex items-center pl-1.5 pr-0.5 py-0 rounded text-[8px] font-medium bg-blue-100 dark:bg-blue-800/50 text-blue-800 dark:text-blue-200">
                              <TagIcon className="h-1.5 w-1.5 mr-0.5 flex-shrink-0" /> {tag}
                              <button
                                onClick={() => handleRemoveTag(tag)}
                                disabled={isTagLoading}
                                className="ml-0.5 opacity-0 group-hover:opacity-100 focus:opacity-100 rounded-full hover:bg-blue-200 dark:hover:bg-blue-700 p-0.5 disabled:opacity-50"
                                aria-label={`Remove tag ${tag}`}
                              >
                                <XMarkIcon className="h-1.5 w-1.5 text-blue-600 dark:text-blue-300" />
                              </button>
                            </span>
                          )) : <p className="text-gray-500 dark:text-gray-400 italic text-[9px]">No tags yet.</p>}
                        </div>
                        
                        {/* Add Tag Input - More compact */}
                        <div className="flex items-center gap-1">
                          <input
                            type="text"
                            value={tagInput}
                            onChange={(e) => setTagInput(e.target.value)}
                            placeholder="Add a tag..."
                            disabled={isTagLoading}
                            onKeyDown={(e) => { if (e.key === 'Enter') handleAddTag(); }}
                            className="flex-grow px-1.5 py-0 border border-gray-300 rounded text-[9px]
                                     bg-white dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300
                                     focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500
                                     disabled:bg-gray-100 dark:disabled:bg-gray-800"
                          />
                          <button
                            onClick={handleAddTag}
                            disabled={isTagLoading || !tagInput.trim()}
                            className="p-0.5 rounded bg-indigo-500 hover:bg-indigo-600 text-white disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed"
                            aria-label="Add tag"
                          >
                            <PlusIcon className={`h-2 w-2 ${isTagLoading ? 'animate-spin' : ''}`} />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {error && <p className="mt-1 text-xs text-red-600 dark:text-red-400">{error}</p>}
                </div>
              </div>
            ) : (
               // Placeholder when no interaction is selected
              <div className="p-4 text-center text-gray-500 dark:text-gray-400 h-full flex items-center justify-center">
                <div className="text-center">
                   <ChatBubbleLeftRightIcon className="mx-auto h-8 w-8 text-gray-400" aria-hidden="true"/>
                  <h3 className="mt-2 text-xs font-medium text-gray-700 dark:text-gray-300">Select an Interaction</h3>
                  <p className="mt-1 text-[10px] text-gray-500 dark:text-gray-400">
                    Choose an item from the list to see its details and manage tags.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export default History;