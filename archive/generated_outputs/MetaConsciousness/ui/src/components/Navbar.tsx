import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  Bars3Icon, // Hamburger
  MagnifyingGlassIcon,
  SunIcon,
  MoonIcon,
  BellIcon,
  UserCircleIcon, // For Profile Dropdown
  Cog6ToothIcon, // For Settings Dropdown
  ArrowRightOnRectangleIcon, // For Logout Dropdown
  ChevronLeftIcon, // For sidebar toggle
  ChevronRightIcon // For sidebar toggle
} from '@heroicons/react/24/outline';

interface NavbarProps {
  toggleSidebar: () => void;
  isSidebarOpen: boolean; // Prop might be used for styling hamburger later
  darkMode: boolean;
  toggleDarkMode: () => void;
  isPermanentlyHidden: boolean;
  togglePermanentSidebar: () => void;
}

// Hook to detect clicks outside an element
function useOutsideAlerter(ref: React.RefObject<HTMLElement>, callback: () => void) {
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        callback(); // Call the callback function if click is outside
      }
    }
    // Bind the event listener
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      // Unbind the event listener on clean up
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [ref, callback]);
}


const Navbar: React.FC<NavbarProps> = ({ toggleSidebar, isSidebarOpen, darkMode, toggleDarkMode, isPermanentlyHidden, togglePermanentSidebar }) => {
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);
  const [isNotificationsMenuOpen, setIsNotificationsMenuOpen] = useState(false);

  // Refs for dropdowns to detect outside clicks
  const profileMenuRef = useRef<HTMLLIElement>(null);
  const notificationsMenuRef = useRef<HTMLLIElement>(null);

  // Close dropdowns when clicking outside
  useOutsideAlerter(profileMenuRef, () => setIsProfileMenuOpen(false));
  useOutsideAlerter(notificationsMenuRef, () => setIsNotificationsMenuOpen(false));


  const toggleProfileMenu = () => {
    setIsProfileMenuOpen(!isProfileMenuOpen);
    if (isNotificationsMenuOpen) setIsNotificationsMenuOpen(false); // Close other menu
  };

  const toggleNotificationsMenu = () => {
    setIsNotificationsMenuOpen(!isNotificationsMenuOpen);
    if (isProfileMenuOpen) setIsProfileMenuOpen(false); // Close other menu
  };

  return (
    // Reduced vertical padding, adjusted shadow
    <header className="z-20 py-2 bg-white shadow-sm dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 sticky top-0">
      {/* Reduced horizontal padding */}
      <div className="container mx-auto flex items-center justify-between h-full px-4">
        <div className="flex items-center">
          {/* Mobile hamburger */}
          <button
            className="p-1.5 -ml-1 rounded-md md:hidden text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1 dark:focus:ring-offset-gray-800"
            onClick={toggleSidebar}
            aria-label="Menu"
          >
            <Bars3Icon className="w-5 h-5" />
          </button>
          
          {/* Desktop sidebar toggle button */}
          <button
            className="p-1.5 ml-1 hidden md:block rounded-md text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1 dark:focus:ring-offset-gray-800"
            onClick={togglePermanentSidebar}
            aria-label={isPermanentlyHidden ? "Show sidebar" : "Hide sidebar"}
          >
            {isPermanentlyHidden ? (
              <ChevronRightIcon className="w-5 h-5" />
            ) : (
              <ChevronLeftIcon className="w-5 h-5" />
            )}
          </button>
        </div>

        {/* Search input (Presentational) */}
        {/* Adjusted padding, text size, border */}
        <div className="flex justify-center flex-1 lg:mr-32 ml-3 md:ml-0"> {/* Added margin for mobile */}
          <div className="relative w-full max-w-md"> {/* Reduced max width */}
            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
               <MagnifyingGlassIcon className="w-4 h-4 text-gray-400" aria-hidden="true"/>
            </div>
            <input
              className="w-full pl-9 pr-4 py-1.5 text-sm border border-gray-300 rounded-md bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 focus:bg-white dark:focus:bg-gray-700"
              type="search"
              placeholder="Search..." // Simplified placeholder
              aria-label="Search"
            />
          </div>
        </div>

        {/* Navbar Icons */}
        <ul className="flex items-center flex-shrink-0 space-x-4">
          {/* Theme toggler */}
          <li className="flex">
            <button
              className="p-1.5 rounded-md text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1 dark:focus:ring-offset-gray-800"
              onClick={toggleDarkMode}
              aria-label="Toggle theme"
            >
               {darkMode ? (
                 <SunIcon className="w-5 h-5" aria-hidden="true" />
               ) : (
                 <MoonIcon className="w-5 h-5" aria-hidden="true" />
               )}
            </button>
          </li>

          {/* Notifications menu */}
          <li className="relative" ref={notificationsMenuRef}>
            <button
              className="relative p-1.5 rounded-md text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1 dark:focus:ring-offset-gray-800"
              onClick={toggleNotificationsMenu}
              aria-label="Notifications"
              aria-haspopup="true"
            >
               <BellIcon className="w-5 h-5" aria-hidden="true"/>
              {/* Notification badge */}
              <span
                aria-hidden="true"
                // Smaller badge, adjusted position
                className="absolute top-1 right-1 block w-1.5 h-1.5 bg-red-500 border border-white rounded-full dark:border-gray-800"
              ></span>
            </button>

            {isNotificationsMenuOpen && (
              // Adjusted position, width, padding, font size
              <div
                className="absolute right-0 w-64 mt-2 origin-top-right bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none"
                role="menu" aria-orientation="vertical" aria-labelledby="notifications-menu-button" tabIndex={-1}
              >
                <div className="py-1">
                  <div className="px-3 py-2 text-xs font-semibold text-gray-700 dark:text-gray-200 border-b dark:border-gray-700">Notifications</div>
                  {/* Mock Notification Items */}
                  <a
                    href="#"
                    className="flex justify-between items-center px-3 py-2 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                    role="menuitem" tabIndex={-1} id="notification-item-1"
                  >
                    <span className="truncate pr-2">Agent 'Research Assistant' is ready</span>
                    <span className="flex-shrink-0 px-1.5 py-0.5 text-[10px] font-bold leading-none text-red-600 bg-red-100 dark:text-red-100 dark:bg-red-600 rounded-full">New</span>
                  </a>
                   <a
                    href="#"
                    className="block px-3 py-2 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                    role="menuitem" tabIndex={-1} id="notification-item-2"
                  >
                    <span className="truncate">Knowledge base updated</span>
                  </a>
                   {/* Add more notifications or empty state */}
                   <div className="px-3 py-2 text-center text-xs text-gray-500 dark:text-gray-400 border-t dark:border-gray-700">
                      <a href="#" className="hover:underline">View all notifications</a>
                   </div>
                </div>
              </div>
            )}
          </li>

          {/* Profile menu */}
          <li className="relative" ref={profileMenuRef}>
            <button
              className="block rounded-full focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-indigo-500 dark:focus:ring-offset-gray-800" // Removed align-middle
              onClick={toggleProfileMenu}
              aria-label="Account menu"
              aria-haspopup="true"
              id="profile-menu-button"
            >
              <img
                className="object-cover w-7 h-7 rounded-full border-2 border-transparent hover:border-indigo-300 dark:hover:border-indigo-600" // Slightly larger, added hover border
                src="https://i.pravatar.cc/40" // Placeholder
                alt="User profile"
                aria-hidden="true"
              />
            </button>

            {isProfileMenuOpen && (
              <div
                 className="absolute right-0 w-48 mt-2 origin-top-right bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none"
                 role="menu" aria-orientation="vertical" aria-labelledby="profile-menu-button" tabIndex={-1}
              >
                <div className="py-1" role="none">
                   {/* Use role="menuitem" for links/buttons inside */}
                   <Link
                    to="/profile" // Assume /profile route exists
                    className="flex items-center w-full px-3 py-1.5 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                    role="menuitem" tabIndex={-1} id="profile-item-0"
                    onClick={() => setIsProfileMenuOpen(false)} // Close menu on click
                  >
                    <UserCircleIcon className="w-4 h-4 mr-2" aria-hidden="true"/>
                    <span>Profile</span>
                  </Link>
                  <Link
                    to="/settings"
                    className="flex items-center w-full px-3 py-1.5 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                    role="menuitem" tabIndex={-1} id="profile-item-1"
                    onClick={() => setIsProfileMenuOpen(false)}
                  >
                     <Cog6ToothIcon className="w-4 h-4 mr-2" aria-hidden="true"/>
                    <span>Settings</span>
                  </Link>
                   <div className="border-t border-gray-100 dark:border-gray-700 my-1" role="separator"></div>
                  <button // Use button for actions like logout
                    // TODO: Implement logout functionality
                    onClick={() => { alert('Logout Clicked!'); setIsProfileMenuOpen(false); }}
                    className="flex items-center w-full px-3 py-1.5 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                    role="menuitem" tabIndex={-1} id="profile-item-2"
                  >
                     <ArrowRightOnRectangleIcon className="w-4 h-4 mr-2" aria-hidden="true"/>
                    <span>Log out</span>
                  </button>
                </div>
              </div>
            )}
          </li>
        </ul>
      </div>
    </header>
  );
};

export default Navbar;