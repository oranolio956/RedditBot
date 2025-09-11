/**
 * Personality Evolution Page
 * Visualization of personality changes over time
 */

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

export default function PersonalityEvolution() {
  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-insight-title font-bold text-gradient mb-8">
        Personality Evolution
      </h1>
      <Card>
        <CardHeader>
          <CardTitle>Personality evolution tracking under construction</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Advanced personality change visualization and analysis coming soon.</p>
        </CardContent>
      </Card>
    </div>
  );
}