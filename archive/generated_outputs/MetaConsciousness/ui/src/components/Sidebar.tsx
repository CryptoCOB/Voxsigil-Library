import React, { useEffect } from 'react';
import { NavLink } from 'react-router-dom';

interface SidebarProps {
  open: boolean;
  setOpen: (open: boolean) => void;
  isPermanentlyHidden: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ open, setOpen, isPermanentlyHidden }) => {

  // Save sidebar state to localStorage
  useEffect(() => {
    localStorage.setItem('sidebarOpen', JSON.stringify(open));
  }, [open]);

  return (
    <>
      {/* Mobile sidebar backdrop */}
      <div
        className={`fixed inset-0 z-10 flex items-end bg-black bg-opacity-50 sm:items-center sm:justify-center transition-opacity duration-150 ease-in-out ${
          open ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={() => setOpen(false)}
      />

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 z-20 flex-shrink-0 w-56 mt-16 overflow-y-auto bg-white dark:bg-gray-800 transition-all duration-300 ease-in-out ${
          open ? 'translate-x-0' : '-translate-x-64'
        } ${isPermanentlyHidden ? 'md:-translate-x-64' : 'md:translate-x-0'}`}
      >
        <div className="py-2 text-gray-500 dark:text-gray-400 text-xs">
          <a
            className="ml-6 text-sm font-bold text-gray-800 dark:text-gray-200"
            href="#"
          >
            MetaConsciousness
          </a>
          <ul className="mt-4">
            <li className="relative px-4 py-2">
              <NavLink
                className={({ isActive }) =>
                  `inline-flex items-center w-full text-xs font-semibold transition-colors duration-150 hover:text-gray-800 dark:hover:text-gray-200 ${
                    isActive
                      ? 'text-gray-800 dark:text-gray-100'
                      : 'text-gray-500 dark:text-gray-400'
                  }`
                }
                to="/"
                end
              >
                {({ isActive }) => (
                  <>
                    {isActive && (
                      <span
                        className="absolute inset-y-0 left-0 w-1 bg-indigo-600 rounded-tr-lg rounded-br-lg"
                        aria-hidden="true"
                      ></span>
                    )}
                    <svg
                      className="w-3.5 h-3.5 mr-2"
                      aria-hidden="true"
                      fill="none"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      viewBox="0 0 264 464"
                      stroke="currentColor"
                    >
                      <path d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path>
                    </svg>
                    <span>Dashboard</span>
                  </>
                )}
              </NavLink>
            </li>
          </ul>
          <ul>
            <li className="relative px-4 py-2">
              <NavLink
                className={({ isActive }) =>
                  `inline-flex items-center w-full text-xs font-semibold transition-colors duration-150 hover:text-gray-800 dark:hover:text-gray-200 ${
                    isActive
                      ? 'text-gray-800 dark:text-gray-100'
                      : 'text-gray-500 dark:text-gray-400'
                  }`
                }
                to="/history"
              >
                {({ isActive }) => (
                  <>
                    {isActive && (
                      <span
                        className="absolute inset-y-0 left-0 w-1 bg-indigo-600 rounded-tr-lg rounded-br-lg"
                        aria-hidden="true"
                      ></span>
                    )}
                    <svg
                      className="w-3.5 h-3.5 mr-2"
                      aria-hidden="true"
                      fill="none"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      viewBox="0 0 264 464"
                      stroke="currentColor"
                    >
                      <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"></path>
                    </svg>
                    <span>History</span>
                  </>
                )}
              </NavLink>
            </li>
            <li className="relative px-4 py-2">
              <NavLink
                className={({ isActive }) =>
                  `inline-flex items-center w-full text-xs font-semibold transition-colors duration-150 hover:text-gray-800 dark:hover:text-gray-200 ${
                    isActive
                      ? 'text-gray-800 dark:text-gray-100'
                      : 'text-gray-500 dark:text-gray-400'
                  }`
                }
                to="/metrics"
              >
                {({ isActive }) => (
                  <>
                    {isActive && (
                      <span
                        className="absolute inset-y-0 left-0 w-1 bg-indigo-600 rounded-tr-lg rounded-br-lg"
                        aria-hidden="true"
                      ></span>
                    )}
                    <svg
                      className="w-3.5 h-3.5 mr-2"
                      aria-hidden="true"
                      fill="none"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      viewBox="0 0 264 464"
                      stroke="currentColor"
                    >
                      <path d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"></path>
                      <path d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"></path>
                    </svg>
                    <span>Metrics</span>
                  </>
                )}
              </NavLink>
            </li>
            <li className="relative px-4 py-2">
              <NavLink
                className={({ isActive }) =>
                  `inline-flex items-center w-full text-xs font-semibold transition-colors duration-150 hover:text-gray-800 dark:hover:text-gray-200 ${
                    isActive
                      ? 'text-gray-800 dark:text-gray-100'
                      : 'text-gray-500 dark:text-gray-400'
                  }`
                }
                to="/models"
              >
                {({ isActive }) => (
                  <>
                    {isActive && (
                      <span
                        className="absolute inset-y-0 left-0 w-1 bg-indigo-600 rounded-tr-lg rounded-br-lg"
                        aria-hidden="true"
                      ></span>
                    )}
                    <svg
                      className="w-3.5 h-3.5 mr-2"
                      aria-hidden="true"
                      fill="none"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      viewBox="0 0 264 464"
                      stroke="currentColor"
                    >
                      <path d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z"></path>
                    </svg>
                    <span>Models</span>
                  </>
                )}
              </NavLink>
            </li>
            <li className="relative px-4 py-2">
              <NavLink
                className={({ isActive }) =>
                  `inline-flex items-center w-full text-xs font-semibold transition-colors duration-150 hover:text-gray-800 dark:hover:text-gray-200 ${
                    isActive
                      ? 'text-gray-800 dark:text-gray-100'
                      : 'text-gray-500 dark:text-gray-400'
                  }`
                }
                to="/settings"
              >
                {({ isActive }) => (
                  <>
                    {isActive && (
                      <span
                        className="absolute inset-y-0 left-0 w-1 bg-indigo-600 rounded-tr-lg rounded-br-lg"
                        aria-hidden="true"
                      ></span>
                    )}
                    <svg
                      className="w-3.5 h-3.5 mr-2"
                      aria-hidden="true"
                      fill="none"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      viewBox="0 0 264 464"
                      stroke="currentColor"
                    >
                      <path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                      <path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                    </svg>
                    <span>Settings</span>
                  </>
                )}
              </NavLink>
            </li>
          </ul>

          <div className="px-4 my-4">
            <button className="flex items-center justify-between w-full px-3 py-1.5 text-xs font-medium leading-5 text-white transition-colors duration-150 bg-indigo-600 border border-transparent rounded-lg active:bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:shadow-outline-purple">
              New Analysis
              <span className="ml-2" aria-hidden="true">+</span>
            </button>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;