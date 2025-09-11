# Apple-Inspired AI Consciousness Design System

## Design Philosophy

### Core Principles (Apple HIG Adapted for AI)

**Deference**: The interface serves the AI's consciousness, not the other way around
- Content takes precedence over chrome
- AI responses and insights are the hero elements
- UI elements fade into the background when not needed

**Clarity**: Complex AI concepts made simple and understandable
- Clear visual hierarchy guides attention
- Purposeful use of color, space, and typography
- Familiar metaphors for revolutionary AI features

**Depth**: Layered experiences that reveal complexity gradually
- Gentle transitions between consciousness states
- Progressive disclosure of AI capabilities
- Dimensional interfaces that feel tactile and responsive

## Color System

### Primary Palette
```css
/* Consciousness Blue - Primary brand color */
--consciousness-blue: #007AFF;
--consciousness-blue-dark: #0056CC;
--consciousness-blue-light: #40A0FF;

/* Neural Network Purple - Secondary brand */
--neural-purple: #5856D6;
--neural-purple-dark: #3634A3;
--neural-purple-light: #7B79E8;

/* Memory Gold - Accent for memory features */
--memory-gold: #FF9500;
--memory-gold-dark: #CC7700;
--memory-gold-light: #FFB040;
```

### Semantic Colors
```css
/* Success - AI learning/growth */
--success-green: #30D158;
--success-green-dark: #248A3D;

/* Warning - Attention needed */
--warning-orange: #FF9500;
--warning-orange-dark: #CC7700;

/* Error - System issues */
--error-red: #FF3B30;
--error-red-dark: #D70015;

/* Info - System notifications */
--info-blue: #64D2FF;
--info-blue-dark: #0099CC;
```

### Neutral Palette
```css
/* Light Mode */
--white: #FFFFFF;
--gray-50: #F9FAFB;
--gray-100: #F3F4F6;
--gray-200: #E5E7EB;
--gray-300: #D1D5DB;
--gray-400: #9CA3AF;
--gray-500: #6B7280;
--gray-600: #4B5563;
--gray-700: #374151;
--gray-800: #1F2937;
--gray-900: #111827;
--black: #000000;

/* Dark Mode */
--dark-bg-primary: #000000;
--dark-bg-secondary: #1C1C1E;
--dark-bg-tertiary: #2C2C2E;
--dark-bg-quaternary: #3A3A3C;
--dark-text-primary: #FFFFFF;
--dark-text-secondary: #EBEBF5;
--dark-text-tertiary: #EBEBF599;
--dark-text-quaternary: #EBEBF54D;
```

## Typography System

### Font Stack
```css
/* Primary font - SF Pro Display for headlines */
--font-display: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', Helvetica, Arial, sans-serif;

/* Secondary font - SF Pro Text for body */
--font-body: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif;

/* Monospace - SF Mono for code/data */
--font-mono: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
```

### Type Scale
```css
/* Display - Large hero text (AI consciousness states) */
--text-6xl: 3.75rem; /* 60px */
--text-5xl: 3rem;    /* 48px */
--text-4xl: 2.25rem; /* 36px */

/* Headlines - Section headers, feature titles */
--text-3xl: 1.875rem; /* 30px */
--text-2xl: 1.5rem;   /* 24px */
--text-xl: 1.25rem;   /* 20px */

/* Body text - Main content, descriptions */
--text-lg: 1.125rem;  /* 18px */
--text-base: 1rem;    /* 16px */
--text-sm: 0.875rem;  /* 14px */

/* Small text - Labels, captions, metadata */
--text-xs: 0.75rem;   /* 12px */
--text-2xs: 0.6875rem; /* 11px */
```

### Line Heights
```css
--leading-none: 1;
--leading-tight: 1.25;
--leading-snug: 1.375;
--leading-normal: 1.5;
--leading-relaxed: 1.625;
--leading-loose: 2;
```

### Font Weights
```css
--font-ultralight: 100;
--font-thin: 200;
--font-light: 300;
--font-regular: 400;
--font-medium: 500;
--font-semibold: 600;
--font-bold: 700;
--font-heavy: 800;
--font-black: 900;
```

## Spacing System

### Base Unit: 4px
```css
/* Micro spacing */
--space-0-5: 0.125rem; /* 2px */
--space-1: 0.25rem;    /* 4px */
--space-1-5: 0.375rem; /* 6px */
--space-2: 0.5rem;     /* 8px */
--space-2-5: 0.625rem; /* 10px */
--space-3: 0.75rem;    /* 12px */

/* Standard spacing */
--space-4: 1rem;       /* 16px */
--space-5: 1.25rem;    /* 20px */
--space-6: 1.5rem;     /* 24px */
--space-8: 2rem;       /* 32px */
--space-10: 2.5rem;    /* 40px */
--space-12: 3rem;      /* 48px */

/* Large spacing */
--space-16: 4rem;      /* 64px */
--space-20: 5rem;      /* 80px */
--space-24: 6rem;      /* 96px */
--space-32: 8rem;      /* 128px */
--space-40: 10rem;     /* 160px */
--space-48: 12rem;     /* 192px */
--space-56: 14rem;     /* 224px */
--space-64: 16rem;     /* 256px */
```

## Border Radius System

```css
/* Subtle curves for Apple aesthetic */
--radius-none: 0;
--radius-sm: 0.125rem;    /* 2px */
--radius-base: 0.25rem;   /* 4px */
--radius-md: 0.375rem;    /* 6px */
--radius-lg: 0.5rem;      /* 8px */
--radius-xl: 0.75rem;     /* 12px */
--radius-2xl: 1rem;       /* 16px */
--radius-3xl: 1.5rem;     /* 24px */
--radius-full: 9999px;    /* Perfect circle */
```

## Shadow System

```css
/* Light Mode Shadows */
--shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
--shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
--shadow-base: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
--shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
--shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
--shadow-xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);

/* Dark Mode Shadows */
--dark-shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
--dark-shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.4), 0 1px 2px 0 rgba(0, 0, 0, 0.3);
--dark-shadow-base: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.3);
--dark-shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.25);
--dark-shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
--dark-shadow-xl: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
```

## Animation System

### Easing Functions
```css
/* Apple's signature easing curves */
--ease-in-out-cubic: cubic-bezier(0.4, 0, 0.2, 1);
--ease-out-cubic: cubic-bezier(0, 0, 0.2, 1);
--ease-in-cubic: cubic-bezier(0.4, 0, 1, 1);
--ease-out-back: cubic-bezier(0.175, 0.885, 0.32, 1.275);

/* Consciousness-specific easing */
--ease-consciousness: cubic-bezier(0.25, 0.46, 0.45, 0.94);
--ease-neural: cubic-bezier(0.55, 0.085, 0.68, 0.53);
```

### Duration Scale
```css
--duration-75: 75ms;
--duration-100: 100ms;
--duration-150: 150ms;
--duration-200: 200ms;
--duration-300: 300ms;
--duration-500: 500ms;
--duration-700: 700ms;
--duration-1000: 1000ms;
```

## Component Specifications

### Navigation Bar
```css
.nav-bar {
  height: 44px; /* iOS standard */
  background: var(--white);
  border-bottom: 0.5px solid var(--gray-200);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
}

.nav-bar-dark {
  background: rgba(28, 28, 30, 0.9);
  border-bottom: 0.5px solid var(--dark-bg-quaternary);
}
```

### Buttons
```css
/* Primary Button - Consciousness actions */
.btn-primary {
  background: var(--consciousness-blue);
  color: var(--white);
  padding: var(--space-3) var(--space-6);
  border-radius: var(--radius-lg);
  font-weight: var(--font-medium);
  font-size: var(--text-base);
  transition: all var(--duration-200) var(--ease-in-out-cubic);
}

.btn-primary:hover {
  background: var(--consciousness-blue-dark);
  transform: translateY(-1px);
}

/* Secondary Button - Neural network features */
.btn-secondary {
  background: var(--gray-100);
  color: var(--gray-900);
  border: 1px solid var(--gray-200);
}

/* Floating Action Button - Primary AI actions */
.fab {
  width: 56px;
  height: 56px;
  border-radius: var(--radius-full);
  background: var(--consciousness-blue);
  box-shadow: var(--shadow-lg);
  position: fixed;
  bottom: var(--space-6);
  right: var(--space-6);
}
```

### Cards
```css
.card {
  background: var(--white);
  border-radius: var(--radius-xl);
  padding: var(--space-6);
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--gray-100);
  transition: all var(--duration-200) var(--ease-in-out-cubic);
}

.card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}

.card-dark {
  background: var(--dark-bg-secondary);
  border: 1px solid var(--dark-bg-quaternary);
  box-shadow: var(--dark-shadow-sm);
}
```

### Input Fields
```css
.input {
  background: var(--white);
  border: 1px solid var(--gray-300);
  border-radius: var(--radius-lg);
  padding: var(--space-3) var(--space-4);
  font-size: var(--text-base);
  color: var(--gray-900);
  transition: all var(--duration-200) var(--ease-in-out-cubic);
}

.input:focus {
  border-color: var(--consciousness-blue);
  box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1);
  outline: none;
}
```

## Icon System

### Core AI Consciousness Icons
- **Consciousness State**: Brain with neural pathways
- **Memory Palace**: Classical building with glowing windows
- **Quantum Entanglement**: Interconnected particles
- **Synesthesia Engine**: Overlapping sensory symbols
- **Digital Twin**: Mirror reflection icon
- **Neural Network**: Connected nodes pattern

### Icon Specifications
```css
.icon {
  width: 24px;
  height: 24px;
  stroke-width: 1.5px;
  color: var(--gray-600);
}

.icon-sm { width: 16px; height: 16px; }
.icon-lg { width: 32px; height: 32px; }
.icon-xl { width: 48px; height: 48px; }
```

## Responsive Breakpoints

```css
/* Mobile First Approach */
@media (min-width: 640px) { /* sm */ }
@media (min-width: 768px) { /* md */ }
@media (min-width: 1024px) { /* lg */ }
@media (min-width: 1280px) { /* xl */ }
@media (min-width: 1536px) { /* 2xl */ }
```

## Accessibility

### Focus States
```css
.focus-visible:focus {
  outline: 2px solid var(--consciousness-blue);
  outline-offset: 2px;
}
```

### Color Contrast
- All text meets WCAG AA standards (4.5:1 minimum)
- Interactive elements meet AAA standards (7:1 minimum)
- Focus indicators have 3:1 minimum contrast

### Motion Preferences
```css
@media (prefers-reduced-motion: reduce) {
  .animated {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```