/**
 * Future Self Page
 * Simulation and conversation with predicted future self
 */

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

export default function FutureSelf() {
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-insight-title font-bold text-gradient mb-8">
        Future Self Simulation
      </h1>
      <Card>
        <CardHeader>
          <CardTitle>Future self simulation under construction</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Conversation with your predicted future consciousness coming soon.</p>
        </CardContent>
      </Card>
    </div>
  );
}