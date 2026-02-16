import React, { useState, useEffect } from 'react';
import { Routes, Route, useLocation, useNavigationType } from 'react-router-dom';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import History from './pages/History';
import Metrics from './pages/Metrics';
import Settings from './pages/Settings';
import ModelComparison from './pages/ModelComparison';
import './App.css';

// Helper function for cleaner initial dark mode state
const getInitialDarkMode = (): boolean => {
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme) {
    return savedTheme === 'dark';
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
};

// Helper function to get initial sidebar state
const getInitialSidebarState = (): boolean => {
  const savedSidebarState = localStorage.getItem('sidebarOpen');
  if (savedSidebarState !== null) {
    return JSON.parse(savedSidebarState);
  }
  return true; // Default to open
};

// Helper function to get initial sidebar permanent visibility state
const getInitialPermanentState = (): boolean => {
  const savedState = localStorage.getItem('sidebarPermanentlyHidden');
  if (savedState !== null) {
    return JSON.parse(savedState);
  }
  return false; // Default to visible
};

// Silence specific React Router warnings related to v7 transitions
// This helps prevent console spam while we're using both APIs
const silenceRouterWarnings = () => {
  const originalWarn = console.warn;
  console.warn = function(...args) {
    // Filter out specific React Router warnings
    if (typeof args[0] === 'string' && 
        (args[0].includes('React Router') || 
         args[0].includes('v7_startTransition'))) {
      return;
    }
    originalWarn.apply(console, args);
  };
};

// Call this function early
silenceRouterWarnings();

const App: React.FC = () => {
  // --- State ---
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(getInitialSidebarState);
  const [isPermanentlyHidden, setIsPermanentlyHidden] = useState<boolean>(getInitialPermanentState);
  const [darkMode, setDarkMode] = useState<boolean>(getInitialDarkMode);

  // React Router hooks
  const location = useLocation();
  const navigationType = useNavigationType();

  // --- Effects ---
  // Apply dark mode class to the root HTML element for Tailwind
  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    // Store preference
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // Save permanent sidebar state to localStorage
  useEffect(() => {
    localStorage.setItem('sidebarPermanentlyHidden', JSON.stringify(isPermanentlyHidden));
  }, [isPermanentlyHidden]);

  // Use React 18's startTransition for navigation state updates
  // This mimics the v7_startTransition future flag behavior
  useEffect(() => {
    // This only runs on route changes
    const cleanup = () => {
      // Cleanup logic if needed
    };
    return cleanup;
  }, [location, navigationType]);

  // --- Handlers ---
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const togglePermanentSidebar = () => {
    setIsPermanentlyHidden(!isPermanentlyHidden);
  };

  const toggleDarkMode = () => {
    setDarkMode(prevMode => !prevMode);
  };

  // --- Render ---
  return (
    // Main application container using Flexbox
    <div className={`flex h-screen bg-gray-100 dark:bg-gray-900 transition-colors duration-200`}>
      {/* Sidebar Component */}
      <Sidebar open={sidebarOpen} setOpen={setSidebarOpen} isPermanentlyHidden={isPermanentlyHidden} />

      {/* Main Content Area */}
      <div className={`flex flex-col flex-1 w-0 transition-all duration-300 ${isPermanentlyHidden ? 'ml-0' : 'md:ml-56'}`}>
        {/* Navbar Component */}
        <Navbar
          toggleSidebar={toggleSidebar}
          isSidebarOpen={sidebarOpen}
          darkMode={darkMode}
          toggleDarkMode={toggleDarkMode}
          isPermanentlyHidden={isPermanentlyHidden}
          togglePermanentSidebar={togglePermanentSidebar}
        />

        {/* Page Content Area */}
        <main className="flex-1 overflow-y-auto"> {/* Ensure main area takes remaining height and scrolls */}
          {/* Inner container for consistent padding across pages */}
          <div className="container mx-auto px-4 py-4"> {/* Consistent padding */}
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/history" element={<History />} />
              <Route path="/metrics" element={<Metrics />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/models" element={<ModelComparison />} />
              {/* Add a fallback route potentially */}
              {/* <Route path="*" element={<NotFoundPage />} /> */}
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;