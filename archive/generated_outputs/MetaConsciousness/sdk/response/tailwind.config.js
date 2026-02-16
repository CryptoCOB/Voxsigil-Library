/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./frontend/src/**/*.{js,jsx,ts,tsx}",
    "./frontend/frontend/src/**/*.{js,jsx,ts,tsx}",
    "./frontend/index.html",
    "./frontend/public/**/*.html",
    "./src/**/*.{js,jsx,ts,tsx}",
    "./index.html"
  ],
  darkMode: 'class', // Add this to ensure dark mode classes work
  theme: {
    extend: {
      fontSize: {
        'xs': '0.75rem',     /* 12px */
        'sm': '0.8125rem',   /* 13px */
        'base': '0.875rem',  /* 14px instead of 16px */
        'lg': '0.9375rem',   /* 15px */
        'xl': '1rem',        /* 16px */
        '2xl': '1.125rem',   /* 18px */
      },
      spacing: {
        '1': '0.25rem',      /* 4px */
        '2': '0.375rem',     /* 6px */
        '3': '0.5rem',       /* 8px */
        '4': '0.75rem',      /* 12px */
        '5': '1rem',         /* 16px */
      }
    },
  },
  plugins: [],
}

