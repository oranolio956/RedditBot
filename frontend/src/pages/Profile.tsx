/**
 * Profile Page
 * User profile and account management
 */

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

export default function Profile() {
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-insight-title font-bold text-gradient mb-8">
        Profile
      </h1>
      <Card>
        <CardHeader>
          <CardTitle>Profile management under construction</CardTitle>
        </CardHeader>
        <CardContent>
          <p>User profile and consciousness identity management coming soon.</p>
        </CardContent>
      </Card>
    </div>
  );
}