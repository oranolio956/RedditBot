/**
 * Authentication Layout Component
 * Minimal layout for login/register pages
 */

import React from 'react';

interface AuthLayoutProps {
  children: React.ReactNode;
}

export default function AuthLayout({ children }: AuthLayoutProps) {
  return (
    <div className="min-h-screen bg-surface-primary">
      {children}
    </div>
  );
}