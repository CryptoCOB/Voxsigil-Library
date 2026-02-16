import React, { useState, useEffect } from 'react';
import modelService from '../utils/modelService'; // Corrected import path

/**
 * Settings component for configuring application settings
 * including Ollama and LMStudio connection settings
 */
const Settings: React.FC = () => {
  // State for model service settings
  const [ollamaUrl, setOllamaUrl] = useState<string>('http://localhost:11434');
  const [lmStudioUrl, setLmStudioUrl] = useState<string>('http://localhost:1234');
  const [savedMessage, setSavedMessage] = useState<string>('');
  const [saveError, setSaveError] = useState<string>('');
  const [ollamaStatus, setOllamaStatus] = useState<string>('checking');
  const [lmStudioStatus, setLmStudioStatus] = useState<string>('checking');

  // Load saved settings from localStorage on component mount
  useEffect(() => {
    const savedOllamaUrl = localStorage.getItem('ollamaUrl');
    const savedLmStudioUrl = localStorage.getItem('lmStudioUrl');

    if (savedOllamaUrl) {
      setOllamaUrl(savedOllamaUrl);
    }

    if (savedLmStudioUrl) {
      setLmStudioUrl(savedLmStudioUrl);
    }

    // Check model services status
    checkServiceStatus();
  }, []);

  // Check status of Ollama and LMStudio services
  const checkServiceStatus = async () => {
    try {
      const status = await modelService.checkStatus();
      setOllamaStatus(status.ollama.status);
      setLmStudioStatus(status.lmstudio.status);
    } catch (err) {
      console.error('Error checking service status:', err);
      setOllamaStatus('error');
      setLmStudioStatus('error');
    }
  };

  // Save settings to localStorage
  const saveSettings = () => {
    try {
      // Validate URLs
      new URL(ollamaUrl);
      new URL(lmStudioUrl);

      // Save to localStorage
      localStorage.setItem('ollamaUrl', ollamaUrl);
      localStorage.setItem('lmStudioUrl', lmStudioUrl);

      // Show success message
      setSavedMessage('Settings saved successfully!');
      setSaveError('');

      // Clear success message after 3 seconds
      setTimeout(() => {
        setSavedMessage('');
      }, 3000);

      // Check services status with new URLs
      checkServiceStatus();
    } catch (err) {
      setSaveError('Please enter valid URLs');
      setSavedMessage('');
    }
  };

  // Reset settings to defaults
  const resetSettings = () => {
    setOllamaUrl('http://localhost:11434');
    setLmStudioUrl('http://localhost:1234');
    setSavedMessage('');
    setSaveError('');
  };

  return (
    <div className="flex flex-col space-y-6">
      <h1 className="text-xl font-semibold text-gray-700 dark:text-gray-200">
        Settings
      </h1>

      {/* Model Service Settings */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-700 dark:text-gray-200">
            Model Services
          </h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Configure connections to Ollama and LMStudio model services.
          </p>
        </div>

        <div className="p-4 space-y-4">
          {/* Ollama Settings */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Ollama API URL
              </label>
              <span className={`inline-block px-2 py-1 text-xs rounded-full ${
                ollamaStatus === 'available'
                  ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                  : ollamaStatus === 'checking'
                    ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                    : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
              }`}>
                {ollamaStatus === 'available' ? 'Available' : 
                 ollamaStatus === 'checking' ? 'Checking...' : 'Unavailable'}
              </span>
            </div>
            <input
              type="text"
              className="w-full px-3 py-2 text-sm bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md focus:ring-1 focus:ring-blue-500 focus:border-blue-500 dark:text-white"
              value={ollamaUrl}
              onChange={(e) => setOllamaUrl(e.target.value)}
              placeholder="http://localhost:11434"
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Default: http://localhost:11434
            </p>
          </div>

          {/* LMStudio Settings */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                LMStudio API URL
              </label>
              <span className={`inline-block px-2 py-1 text-xs rounded-full ${
                lmStudioStatus === 'available'
                  ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                  : lmStudioStatus === 'checking'
                    ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                    : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
              }`}>
                {lmStudioStatus === 'available' ? 'Available' : 
                 lmStudioStatus === 'checking' ? 'Checking...' : 'Unavailable'}
              </span>
            </div>
            <input
              type="text"
              className="w-full px-3 py-2 text-sm bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md focus:ring-1 focus:ring-blue-500 focus:border-blue-500 dark:text-white"
              value={lmStudioUrl}
              onChange={(e) => setLmStudioUrl(e.target.value)}
              placeholder="http://localhost:1234"
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Default: http://localhost:1234
            </p>
          </div>

          {/* Status Messages */}
          {savedMessage && (
            <div className="p-2 bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 text-sm rounded">
              {savedMessage}
            </div>
          )}
          
          {saveError && (
            <div className="p-2 bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 text-sm rounded">
              {saveError}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-end space-x-3 pt-2">
            <button
              type="button"
              onClick={resetSettings}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 dark:bg-gray-700 dark:text-gray-300 dark:border-gray-600 dark:hover:bg-gray-600"
            >
              Reset to Default
            </button>
            <button
              type="button"
              onClick={checkServiceStatus}
              className="px-4 py-2 text-sm font-medium text-indigo-700 bg-indigo-100 border border-transparent rounded-md hover:bg-indigo-200 dark:bg-indigo-900 dark:text-indigo-200 dark:hover:bg-indigo-800"
            >
              Test Connection
            </button>
            <button
              type="button"
              onClick={saveSettings}
              className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:ring-offset-gray-800"
            >
              Save Settings
            </button>
          </div>
        </div>
      </div>

      {/* Application Settings */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-700 dark:text-gray-200">
            Application Settings
          </h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Configure application behavior and appearance.
          </p>
        </div>

        {/* Placeholder for additional settings */}
        <div className="p-4">
          <p className="text-sm text-gray-500 dark:text-gray-400 italic">
            Additional application settings will be implemented here.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Settings;