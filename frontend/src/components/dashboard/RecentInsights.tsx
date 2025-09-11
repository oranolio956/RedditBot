/**
 * Recent Insights Component
 * Displays latest AI-generated consciousness insights
 */

import { motion } from 'framer-motion';
import {
  LightBulbIcon,
  SparklesIcon,
  ArrowTrendingUpIcon,
  ClockIcon,
  EyeIcon,
} from '@heroicons/react/24/outline';

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { formatRelativeTime } from '@/lib/utils';

// Mock insights data
const recentInsights = [
  {
    id: '1',
    title: 'Creativity Peak Detected',
    content: 'Your creative thinking shows a consistent peak at 2:30 PM across 15 days of analysis. Consider scheduling creative work during this window.',
    confidence: 0.94,
    type: 'pattern',
    category: 'Productivity',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), // 2 hours ago
    icon: SparklesIcon,
    color: 'consciousness-accent',
  },
  {
    id: '2',
    title: 'Mood-Keystroke Correlation',
    content: 'Strong correlation (r=0.87) detected between emotional state and typing rhythm patterns. Your mood significantly influences digital behavior.',
    confidence: 0.87,
    type: 'correlation',
    category: 'Emotional Intelligence',
    timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(), // 5 hours ago
    icon: ArrowTrendingUpIcon,
    color: 'pink-500',
  },
  {
    id: '3',
    title: 'Decision Speed Improvement',
    content: 'Your decision-making speed increases by 23% during measured flow states. Flow sessions correlate with enhanced cognitive performance.',
    confidence: 0.91,
    type: 'performance',
    category: 'Cognitive Enhancement',
    timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(), // 1 day ago
    icon: LightBulbIcon,
    color: 'consciousness-primary',
  },
  {
    id: '4',
    title: 'Personality Trait Evolution',
    content: 'Openness to experience shows +8% increase over the past month, particularly when discussing technical topics.',
    confidence: 0.83,
    type: 'evolution',
    category: 'Personality',
    timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), // 2 days ago
    icon: EyeIcon,
    color: 'consciousness-secondary',
  },
];

const getInsightTypeColor = (type: string) => {
  switch (type) {
    case 'pattern':
      return 'bg-consciousness-accent/10 text-consciousness-accent';
    case 'correlation':
      return 'bg-pink-100 text-pink-600';
    case 'performance':
      return 'bg-consciousness-primary/10 text-consciousness-primary';
    case 'evolution':
      return 'bg-consciousness-secondary/10 text-consciousness-secondary';
    default:
      return 'bg-gray-100 text-gray-600';
  }
};

export default function RecentInsights() {
  const handleViewInsight = (insightId: string) => {
    console.log('Viewing insight:', insightId);
    // Navigate to detailed insight view
  };

  const handleGenerateNew = () => {
    console.log('Generating new insights...');
    // Trigger new insight generation
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <LightBulbIcon className="w-5 h-5" />
            <span>Recent Insights</span>
          </CardTitle>
          <Button variant="ghost" size="sm" onClick={handleGenerateNew}>
            <SparklesIcon className="w-4 h-4 mr-1" />
            Generate New
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {recentInsights.map((insight, index) => (
            <motion.div
              key={insight.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="group"
            >
              <div
                onClick={() => handleViewInsight(insight.id)}
                className="p-4 border border-gray-200 rounded-lg hover:border-consciousness-primary/30 hover:shadow-card transition-all cursor-pointer"
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className={`w-8 h-8 rounded-lg bg-${insight.color}/10 flex items-center justify-center`}>
                      <insight.icon className={`w-4 h-4 text-${insight.color}`} />
                    </div>
                    <div>
                      <h3 className="font-medium text-text-primary group-hover:text-consciousness-primary transition-colors">
                        {insight.title}
                      </h3>
                      <div className="flex items-center space-x-2 mt-1">
                        <span className={`px-2 py-0.5 text-xs rounded-full font-medium ${getInsightTypeColor(insight.type)}`}>
                          {insight.category}
                        </span>
                        <span className="text-xs text-text-tertiary">
                          {Math.round(insight.confidence * 100)}% confident
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-1 text-text-tertiary">
                    <ClockIcon className="w-3 h-3" />
                    <span className="text-xs">
                      {formatRelativeTime(insight.timestamp)}
                    </span>
                  </div>
                </div>

                {/* Content */}
                <p className="text-sm text-text-secondary leading-relaxed">
                  {insight.content}
                </p>

                {/* Confidence bar */}
                <div className="mt-3 flex items-center space-x-2">
                  <span className="text-xs text-text-tertiary">Confidence:</span>
                  <div className="flex-1 bg-gray-200 rounded-full h-1">
                    <div
                      className="h-1 bg-consciousness-primary rounded-full transition-all duration-300"
                      style={{ width: `${insight.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-medium text-text-primary">
                    {Math.round(insight.confidence * 100)}%
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Footer */}
        <div className="mt-6 pt-4 border-t border-gray-200 text-center">
          <Button variant="outline" size="sm">
            View All Insights
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}