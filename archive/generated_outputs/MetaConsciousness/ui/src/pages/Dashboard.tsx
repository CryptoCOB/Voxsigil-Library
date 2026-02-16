import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios'; // Import axios
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
  ChartData
} from 'chart.js';
import {
  ChartBarIcon,
  ClockIcon,
  UsersIcon,
  CpuChipIcon,
  ExclamationTriangleIcon, // For error display
  TagIcon // For history table
} from '@heroicons/react/24/outline';

// Register ChartJS components
ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend
);

// --- Configuration ---
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const RECENT_INTERACTIONS_LIMIT = 5;

// --- Type Definitions (Assume these match backend/shared types) ---

interface TokenData {
    prompt_tokens: number | null;
    completion_tokens: number | null;
    total_tokens: number | null;
}

interface Metadata {
    latency_seconds: number | null;
    tokens: TokenData | null;
}

// Interface for interaction history entries fetched from API
interface InteractionEntry {
  id: string;
  query: string | any; // Adjust based on actual prompt type
  response: string | null;
  model: string;
  source: string;
  type: string;
  tags: string[];
  created_at: string; // ISO string
  metadata: Metadata;
}

// Interface for aggregated metrics from the backend (/api/metrics)
interface MetricSummary {
    name: string;
    call_count: number;
    success_count: number;
    error_count: number;
    avg_latency_seconds: number;
    total_tokens: number | null;
}
interface MetricSummaryResponse {
    time_period: string;
    start_time_utc: string | null;
    end_time_utc: string;
    overall_calls: number;
    overall_success: number;
    overall_errors: number;
    overall_avg_latency: number;
    overall_total_tokens: number | null;
    metrics_by_source: MetricSummary[];
    metrics_by_model: MetricSummary[];
}

// Interface for dashboard-specific stats derived from metrics
interface DashboardStats {
    totalInteractions: number;
    avgResponseTime: number;
    modelCount: number;
    activeUsers: number; // Keep mock or fetch separately if needed
}

// Interface for API responses with results and total_found
interface HistoryResponse {
  results: InteractionEntry[];
  total_found: number;
}

// --- Chart Options (Keep refined options) ---
const lineChartOptions: ChartOptions<'line'> = { /* ... keep as before ... */ };
const barChartOptions: ChartOptions<'bar'> = { /* ... keep as before ... */ };
const capabilityChartOptions: ChartOptions<'bar'> = { /* ... keep as before ... */ };

// Mock Capability Data (keep mock unless fetched from API)
const MOCK_CAPABILITY_DATA: ChartData<'bar'> = {
  labels: ['Reasoning', 'Knowledge', 'Planning', 'Creativity', 'Self-reflect'],
  datasets: [
    { label: 'GPT-5', data: [92, 95, 88, 90, 86], backgroundColor: 'rgba(79, 70, 229, 0.6)', /* ... */ },
    { label: 'Claude-3', data: [90, 92, 89, 85, 92], backgroundColor: 'rgba(249, 115, 22, 0.6)', /* ... */ },
  ],
};
// Mock Performance Data (Placeholder - Replace if fetching real data)
const MOCK_PERFORMANCE_DATA: ChartData<'line'> = {
  labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
  datasets: [
    { label: 'Accuracy (%)', data: [85, 87, 88, 90, 92, 94], /* ... */ yAxisID: 'yAccuracy', },
    { label: 'Response Time (s)', data: [2.1, 1.8, 1.6, 1.5, 1.3, 1.2], /* ... */ yAxisID: 'yResponseTime', },
  ],
};


// --- Helper to format dates ---
const formatDateTime = (isoString: string) => {
  try {
    const date = new Date(isoString);
    return date.toLocaleDateString(undefined, { year: '2-digit', month: 'numeric', day: 'numeric' }) + ' ' +
           date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
  } catch (e) {
    return isoString; // Fallback
  }
};

// --- Component ---
const Dashboard: React.FC = () => {
  // --- State for fetched data ---
  const [statsData, setStatsData] = useState<DashboardStats | null>(null);
  const [performanceChartData, setPerformanceChartData] = useState<ChartData<'line'>>(MOCK_PERFORMANCE_DATA); // Default to mock
  const [modelUsageChartData, setModelUsageChartData] = useState<ChartData<'bar'>>({ labels: [], datasets: [] });
  const [recentInteractions, setRecentInteractions] = useState<InteractionEntry[]>([]);
  const [capabilityChartData] = useState<ChartData<'bar'>>(MOCK_CAPABILITY_DATA); // Keep mock for now

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // --- Data Fetching ---
  const fetchDashboardData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    console.log("Fetching dashboard data...");

    try {
      // Use Promise.all to fetch data concurrently
      const [metricsResponse, historyResponse /*, performanceResponse (if endpoint exists)*/] = await Promise.all([
        axios.get<MetricSummaryResponse>(`${API_BASE_URL}/api/metrics?time_period=all`), // Get overall metrics
        axios.get<HistoryResponse>(`${API_BASE_URL}/api/history`, { // Now expecting the new response type
            params: { limit: RECENT_INTERACTIONS_LIMIT, offset: 0 }
        }),
        // Example: Fetch performance data if endpoint exists
        // axios.get<ChartData<'line'>>(`${API_BASE_URL}/api/metrics/performance?period=6m`),
      ]);

      // 1. Process Metrics Data for Stats Cards & Model Usage Chart
      if (metricsResponse.data) {
        const metrics = metricsResponse.data;
        console.log("Metrics data received:", metrics);

        setStatsData({
          totalInteractions: metrics.overall_calls,
          avgResponseTime: metrics.overall_avg_latency,
          modelCount: metrics.metrics_by_model.length,
          activeUsers: 12, // Keep mock for Active Users or fetch separately
        });

        // Prepare Model Usage Chart Data
        setModelUsageChartData({
            labels: metrics.metrics_by_model.map(m => m.name),
            datasets: [{
                label: 'Interaction Count',
                data: metrics.metrics_by_model.map(m => m.call_count),
                 backgroundColor: [ // Regenerate colors or use a fixed palette based on count
                    'rgba(79, 70, 229, 0.8)', 'rgba(249, 115, 22, 0.8)', 'rgba(20, 184, 166, 0.8)',
                    'rgba(219, 39, 119, 0.8)', 'rgba(59, 130, 246, 0.8)', 'rgba(234, 179, 8, 0.8)',
                 ].slice(0, metrics.metrics_by_model.length), // Slice to match number of models
                 borderWidth: 0, // Remove border for cleaner look?
            }]
        });
      } else {
         console.warn("Metrics response data is missing or invalid.");
         // Set default/empty state for stats and model usage
         setStatsData({ totalInteractions: 0, avgResponseTime: 0, modelCount: 0, activeUsers: 0 });
         setModelUsageChartData({ labels: [], datasets: [] });
      }

      // 2. Process History Data - Updated for new response format
      if (historyResponse.data && historyResponse.data.results) {
           const interactionsData = historyResponse.data.results;
           console.log("History data received:", interactionsData);
           setRecentInteractions(interactionsData);
      } else {
           console.warn("History response data is missing or not in expected format.");
           setRecentInteractions([]);
      }


      // 3. Process Performance Data (if fetched)
      // if (performanceResponse?.data) {
      //   setPerformanceChartData(performanceResponse.data);
      // }

      console.log("Dashboard data fetched successfully.");

    } catch (err: any) {
      console.error("Failed to fetch dashboard data:", err);
      setError(err.response?.data?.detail || err.message || 'Failed to load dashboard data.');
    } finally {
      setIsLoading(false);
    }
  }, []); // No dependencies, fetch once on mount

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  // Helper to display prompt (string or messages)
  const renderPrompt = (query: InteractionEntry['query']): string => {
      if (typeof query === 'string') { return query; }
      if (Array.isArray(query)) { return query.map(msg => msg?.content || '').join(' / ') || 'N/A'; }
      return 'N/A';
  };

  // --- Render ---
  return (
    // Container padding provided by App.tsx
    <div className="max-w-full">
      {/* Use consistent h1 heading with reduced size */}
      <h1 className="mb-3 text-lg font-semibold text-gray-700 dark:text-gray-200">
        Dashboard Overview
      </h1>

      {/* Loading State */}
      {isLoading && (
        <div className="flex justify-center items-center py-8">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-500"></div>
          <p className="ml-3 text-sm text-gray-500 dark:text-gray-400">Loading dashboard data...</p>
        </div>
      )}

      {/* Error State */}
      {error && !isLoading && (
         <div className="p-3 mb-3 bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-700 text-red-700 dark:text-red-300 rounded-lg flex items-center">
            <ExclamationTriangleIcon className="h-4 w-4 mr-2 flex-shrink-0" aria-hidden="true" />
            <p className="text-xs">{error}</p>
         </div>
      )}

      {/* Main Content (only render if not loading and no critical error maybe?) */}
      {!isLoading && !error && statsData && (
        <>
           {/* Stats Cards Grid - More compact with smaller gap */}
            <div className="grid gap-2 mb-3 md:grid-cols-2 xl:grid-cols-4">
                {/* Card: Total Interactions - More compact */}
                <div className="flex items-center p-2 bg-white rounded-lg shadow dark:bg-gray-800">
                  <div className="p-1.5 mr-2 text-indigo-500 bg-indigo-100 rounded-full dark:text-indigo-100 dark:bg-indigo-500">
                    <ChartBarIcon className="w-3.5 h-3.5" />
                  </div>
                  <div>
                    <p className="mb-0.5 text-[10px] font-medium text-gray-600 dark:text-gray-400">Total Interactions</p>
                    <p className="text-sm font-semibold text-gray-700 dark:text-gray-200">{statsData.totalInteractions.toLocaleString()}</p>
                  </div>
                </div>
                {/* Card: Average Response Time */}
                <div className="flex items-center p-2 bg-white rounded-lg shadow dark:bg-gray-800">
                  <div className="p-1.5 mr-2 text-orange-500 bg-orange-100 rounded-full dark:text-orange-100 dark:bg-orange-500">
                    <ClockIcon className="w-3.5 h-3.5" />
                  </div>
                  <div>
                    <p className="mb-0.5 text-[10px] font-medium text-gray-600 dark:text-gray-400">Avg. Response Time</p>
                    <p className="text-sm font-semibold text-gray-700 dark:text-gray-200">{statsData.avgResponseTime.toFixed(1)}s</p>
                  </div>
                </div>
                {/* Card: Models Used */}
                <div className="flex items-center p-2 bg-white rounded-lg shadow dark:bg-gray-800">
                   <div className="p-1.5 mr-2 text-teal-500 bg-teal-100 rounded-full dark:text-teal-100 dark:bg-teal-500">
                    <CpuChipIcon className="w-3.5 h-3.5" />
                  </div>
                  <div>
                    <p className="mb-0.5 text-[10px] font-medium text-gray-600 dark:text-gray-400">Models Used</p>
                    <p className="text-sm font-semibold text-gray-700 dark:text-gray-200">{statsData.modelCount}</p>
                  </div>
                </div>
                {/* Card: Active Users */}
                <div className="flex items-center p-2 bg-white rounded-lg shadow dark:bg-gray-800">
                  <div className="p-1.5 mr-2 text-blue-500 bg-blue-100 rounded-full dark:text-blue-100 dark:bg-blue-500">
                    <UsersIcon className="w-3.5 h-3.5" />
                  </div>
                  <div>
                    <p className="mb-0.5 text-[10px] font-medium text-gray-600 dark:text-gray-400">Active Users</p>
                    <p className="text-sm font-semibold text-gray-700 dark:text-gray-200">{statsData.activeUsers} <span className="text-[9px] text-gray-500">(Mock)</span></p>
                  </div>
                </div>
            </div>

           {/* Charts Grid */}
            <div className="grid gap-3 mb-3 md:grid-cols-1 lg:grid-cols-2">
                {/* Performance Metrics Chart - More compact */}
                <div className="min-w-0 p-2 bg-white rounded-lg shadow dark:bg-gray-800">
                  <h2 className="mb-1 text-xs font-semibold text-gray-800 dark:text-gray-300">Performance Trend (Mock)</h2>
                  <div className="h-56"> {/* Slightly reduced height */}
                    <Line data={performanceChartData} options={lineChartOptions} />
                  </div>
                </div>
                {/* Model Usage Chart - More compact */}
                <div className="min-w-0 p-2 bg-white rounded-lg shadow dark:bg-gray-800">
                  <h2 className="mb-1 text-xs font-semibold text-gray-800 dark:text-gray-300">Model Usage</h2>
                  <div className="h-56">
                     {modelUsageChartData.labels && modelUsageChartData.labels.length > 0 ? (
                       <Bar data={modelUsageChartData} options={barChartOptions} />
                     ) : (
                        <p className="text-center text-[10px] text-gray-500 pt-10">No model usage data available.</p>
                     )}
                  </div>
                </div>
            </div>

            {/* Table and Capability Chart Grid */}
            <div className="grid gap-3 md:grid-cols-1 lg:grid-cols-2">
                 {/* Recent Interactions Table */}
                 <div className="min-w-0 p-2 bg-white rounded-lg shadow dark:bg-gray-800 overflow-hidden">
                   <h2 className="mb-1 text-xs font-semibold text-gray-800 dark:text-gray-300">Recent Interactions</h2>
                   <div className="w-full overflow-x-auto">
                     <table className="w-full whitespace-nowrap">
                       <thead>
                         <tr className="text-[9px] font-semibold tracking-wide text-left text-gray-500 uppercase border-b dark:border-gray-700 bg-gray-50 dark:text-gray-400 dark:bg-gray-900/50">
                           <th className="px-2 py-1.5">Query</th>
                           <th className="px-2 py-1.5">Model</th>
                           <th className="px-2 py-1.5">Tags</th>
                           <th className="px-2 py-1.5 text-right">Latency</th>
                         </tr>
                       </thead>
                       <tbody className="bg-white divide-y dark:divide-gray-700 dark:bg-gray-800">
                         {recentInteractions.length > 0 ? recentInteractions.map((interaction) => (
                           <tr key={interaction.id} className="text-gray-700 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700/50">
                               <td className="px-2 py-1.5">
                                 <p className="text-[10px] font-medium truncate max-w-[180px] sm:max-w-xs" title={renderPrompt(interaction.query)}>
                                    {renderPrompt(interaction.query)}
                                 </p>
                                  <p className="text-[8px] text-gray-500 dark:text-gray-400 mt-0.5">
                                     {formatDateTime(interaction.created_at)}
                                  </p>
                               </td>
                               <td className="px-2 py-1.5 text-[10px]">
                                 <span className="px-1.5 py-0.5 font-medium leading-none rounded-full bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-100">
                                     {interaction.model}
                                 </span>
                               </td>
                                <td className="px-2 py-1.5 text-[9px]">
                                    {interaction.tags && interaction.tags.length > 0 ? (
                                        <div className="flex flex-wrap gap-0.5">
                                            {interaction.tags.slice(0, 2).map(tag => ( // Show max 2 tags here
                                                <span key={tag} className="inline-flex items-center px-1 py-0 rounded text-[8px] font-medium bg-blue-100 dark:bg-blue-800/50 text-blue-700 dark:text-blue-200">
                                                <TagIcon className="h-1.5 w-1.5 mr-0.5" /> {tag}
                                                </span>
                                            ))}
                                            {interaction.tags.length > 2 && <span className="text-[8px] text-gray-400">+{interaction.tags.length - 2}</span>}
                                        </div>
                                    ) : <span className="text-gray-400 italic text-[8px]">None</span>}
                                </td>
                               <td className="px-2 py-1.5 text-[10px] text-right">
                                 {interaction.metadata?.latency_seconds?.toFixed(1) ?? 'N/A'}s
                               </td>
                           </tr>
                         )) : (
                            <tr>
                                <td colSpan={4} className="text-center py-3 text-[10px] text-gray-500 dark:text-gray-400">
                                    No recent interactions found.
                                </td>
                            </tr>
                         )}
                       </tbody>
                     </table>
                   </div>
                    {/* Optional Link to full history */}
                    {recentInteractions.length > 0 && (
                        <div className="pt-1 text-right text-[9px]">
                            <a href="/history" className="text-indigo-600 dark:text-indigo-400 hover:underline">View All History</a>
                        </div>
                    )}
                 </div>

                {/* Capability Comparison Chart (Mock Data) */}
                <div className="min-w-0 p-2 bg-white rounded-lg shadow dark:bg-gray-800">
                  <h2 className="mb-1 text-xs font-semibold text-gray-800 dark:text-gray-300">Capability Comparison (Mock)</h2>
                   <div className="h-56">
                    <Bar data={capabilityChartData} options={capabilityChartOptions} />
                  </div>
                </div>
            </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;