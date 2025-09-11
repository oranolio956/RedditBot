/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Apple-inspired consciousness color system
        consciousness: {
          primary: '#007AFF',
          secondary: '#5856D6',
          accent: '#FF9500',
        },
        states: {
          flow: '#30D158',
          stress: '#FF453A',
          neutral: '#8E8E93',
        },
        surface: {
          primary: '#FFFFFF',
          secondary: '#F2F2F7',
          elevated: 'rgba(255,255,255,0.8)',
          'primary-dark': '#000000',
          'secondary-dark': '#1C1C1E',
        },
        text: {
          primary: '#000000',
          secondary: '#3C3C43',
          tertiary: '#8E8E93',
          'primary-dark': '#FFFFFF',
          'secondary-dark': '#EBEBF5',
          'tertiary-dark': '#8E8E93',
        },
        // Semantic colors
        success: '#30D158',
        warning: '#FF9500',
        error: '#FF453A',
        info: '#007AFF',
      },
      fontFamily: {
        'sf-pro': ['-apple-system', 'BlinkMacSystemFont', 'SF Pro Display', 'SF Pro Text', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        'insight-title': ['28px', { lineHeight: '34px', letterSpacing: '0.36px', fontWeight: '700' }],
        'insight-subtitle': ['22px', { lineHeight: '28px', letterSpacing: '0.35px', fontWeight: '600' }],
        'metric-value': ['34px', { lineHeight: '41px', fontWeight: '300' }],
        'body-text': ['17px', { lineHeight: '22px', letterSpacing: '-0.41px', fontWeight: '400' }],
        'caption-text': ['13px', { lineHeight: '18px', letterSpacing: '-0.08px', fontWeight: '400' }],
      },
      spacing: {
        'xs': '4px',
        'sm': '8px',
        'md': '16px',
        'lg': '24px',
        'xl': '32px',
      },
      borderRadius: {
        'sm': '4px',
        'md': '8px',
        'lg': '12px',
        'xl': '16px',
      },
      boxShadow: {
        'card': '0 2px 8px rgba(0,0,0,0.1)',
        'elevated': '0 4px 16px rgba(0,0,0,0.15)',
        'dramatic': '0 8px 32px rgba(0,0,0,0.2)',
        'consciousness': '0 0 20px rgba(0, 122, 255, 0.3)',
        'glow': '0 0 40px rgba(0, 122, 255, 0.2)',
      },
      animation: {
        'breathing': 'breathing 4s ease-in-out infinite',
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'flow-wave': 'flow-wave 3s ease-in-out infinite',
        'pattern-emerge': 'pattern-emerge 1.5s ease-out',
        'insight-arrive': 'insight-arrive 0.8s ease-out',
        'processing-dots': 'processing-dots 1.5s ease-in-out infinite',
      },
      keyframes: {
        breathing: {
          '0%, 100%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.05)' },
        },
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 20px rgba(0, 122, 255, 0.3)' },
          '50%': { boxShadow: '0 0 40px rgba(0, 122, 255, 0.6)' },
        },
        'flow-wave': {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-4px)' },
        },
        'pattern-emerge': {
          '0%': { opacity: '0', transform: 'scale(0.8)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        'insight-arrive': {
          '0%': { opacity: '0', transform: 'translateY(20px) scale(0.95)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
        'processing-dots': {
          '0%': { opacity: '0.3' },
          '50%': { opacity: '1' },
          '100%': { opacity: '0.3' },
        },
      },
      backdropBlur: {
        'xs': '2px',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}