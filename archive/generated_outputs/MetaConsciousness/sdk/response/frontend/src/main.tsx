import React from 'react';
import ReactDOM from 'react-dom/client';
import {
  createBrowserRouter,
  RouterProvider,
  // Outlet, // Import Outlet if used in ErrorBoundary
  // useRouteError // Import useRouteError if used in ErrorBoundary
} from 'react-router-dom';

// Import utility modules to handle browser extension conflicts and CSS issues
import './utils/extensionGuard';
import './utils/cssImportFixer';

// Import CSS files in the correct order
import './reset.css'; // Reset CSS first
import './index.css'; // Base styles and Tailwind directives
import './components.css'; // Component-specific styles using @apply

// Import the main App component
import App from './App.tsx';

// Configure the browser router
const router = createBrowserRouter(
  [
    {
      path: "*", // Catch-all route: delegates all path handling to the <App> component
      element: <App />,
      // Optional: Add an error boundary for router-level errors
      // errorElement: <ErrorBoundary />,
    }
  ],
  {
    // Opt-in to future features for potential improvements/fixes
    future: {
      v7_relativeSplatPath: true,
      v7_normalizeFormMethod: true,
      v7_startTransition: true, // Add this flag to opt into the v7 behavior early
      // Add other future flags as needed/recommended by react-router updates
    }
  }
);

// Get the root element
const rootElement = document.getElementById('root');

// Render the application
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      {/* Provide the router configuration to the application */}
      <RouterProvider router={router} />
    </React.StrictMode>
  );
} else {
  console.error("Failed to find the root element with ID 'root'.");
}