/**
 * Feature Grid Component
 * Grid of AI consciousness features with quick access
 */

import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  CpuChipIcon,
  BuildingOfficeIcon,
  HeartIcon,
  LinkIcon,
  SwatchIcon,
  CloudIcon,
  EyeIcon,
  SparklesIcon,
  ArrowRightIcon,
} from '@heroicons/react/24/outline';

const features = [
  {
    name: 'Consciousness Mirror',
    description: 'Chat with your digital twin and explore personality patterns',
    icon: CpuChipIcon,
    href: '/consciousness',
    color: 'consciousness',
    status: 'Active',
    metrics: '94% accuracy',
  },
  {
    name: 'Memory Palace',
    description: '3D spatial memory organization system',
    icon: BuildingOfficeIcon,
    href: '/memory',
    color: 'memory',
    status: 'Ready',
    metrics: '156 memories',
  },
  {
    name: 'Emotional Intelligence',
    description: 'AI-powered emotional state tracking and insights',
    icon: HeartIcon,
    href: '/emotional',
    color: 'emotional',
    status: 'Monitoring',
    metrics: 'Balanced mood',
  },
  {
    name: 'Quantum Network',
    description: 'Consciousness entanglement and thought teleportation',
    icon: LinkIcon,
    href: '/quantum',
    color: 'quantum',
    status: 'Connected',
    metrics: '5 entanglements',
  },
  {
    name: 'Digital Synesthesia',
    description: 'Cross-sensory experience conversion engine',
    icon: SwatchIcon,
    href: '/synesthesia',
    color: 'synesthesia',
    status: 'Ready',
    metrics: '23 mappings',
  },
  {
    name: 'Neural Dreams',
    description: 'Dream analysis and interpretation system',
    icon: CloudIcon,
    href: '/dreams',
    color: 'dreams',
    status: 'Active',
    metrics: '18 dreams',
  },
  {
    name: 'Temporal Archaeology',
    description: 'Conversation pattern analysis across time',
    icon: EyeIcon,
    href: '/archaeology',
    color: 'archaeology',
    status: 'Analyzing',
    metrics: '12 patterns',
  },
  {
    name: 'Meta Reality',
    description: 'Reality layer manipulation and perception filters',
    icon: SparklesIcon,
    href: '/meta-reality',
    color: 'meta',
    status: 'Experimental',
    metrics: '3 layers',
  },
];

const getFeatureColor = (color: string) => {
  const colors = {
    consciousness: 'from-consciousness-primary to-consciousness-secondary',
    memory: 'from-purple-500 to-purple-700',
    emotional: 'from-pink-500 to-red-500',
    quantum: 'from-consciousness-secondary to-purple-600',
    synesthesia: 'from-green-400 to-blue-500',
    dreams: 'from-indigo-500 to-purple-600',
    archaeology: 'from-yellow-500 to-orange-500',
    meta: 'from-gray-500 to-gray-700',
  };
  return colors[color as keyof typeof colors] || colors.consciousness;
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'Active':
    case 'Connected':
    case 'Monitoring':
      return 'text-states-flow bg-states-flow/10';
    case 'Ready':
      return 'text-consciousness-primary bg-consciousness-primary/10';
    case 'Analyzing':
      return 'text-consciousness-accent bg-consciousness-accent/10';
    case 'Experimental':
      return 'text-states-neutral bg-states-neutral/10';
    default:
      return 'text-text-tertiary bg-surface-secondary';
  }
};

export default function FeatureGrid() {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-insight-subtitle font-semibold text-text-primary mb-2">
          Revolutionary AI Features
        </h2>
        <p className="text-body-text text-text-secondary max-w-2xl mx-auto">
          Explore cutting-edge consciousness technologies that push the boundaries of AI interaction
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {features.map((feature, index) => (
          <motion.div
            key={feature.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ y: -4 }}
            className="group"
          >
            <Link to={feature.href}>
              <div className="bg-surface-primary rounded-xl border border-gray-200 p-6 hover:shadow-elevated transition-all duration-300 group-hover:border-consciousness-primary/30">
                {/* Icon and status */}
                <div className="flex items-center justify-between mb-4">
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${getFeatureColor(feature.color)} flex items-center justify-center group-hover:scale-110 transition-transform`}>
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <span className={`px-2 py-1 text-xs rounded-full font-medium ${getStatusColor(feature.status)}`}>
                    {feature.status}
                  </span>
                </div>

                {/* Content */}
                <div className="space-y-3">
                  <div>
                    <h3 className="font-semibold text-text-primary group-hover:text-consciousness-primary transition-colors">
                      {feature.name}
                    </h3>
                    <p className="text-sm text-text-secondary mt-1 line-clamp-2">
                      {feature.description}
                    </p>
                  </div>

                  {/* Metrics */}
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-tertiary">
                      {feature.metrics}
                    </span>
                    <ArrowRightIcon className="w-4 h-4 text-consciousness-primary opacity-0 group-hover:opacity-100 transform translate-x-0 group-hover:translate-x-1 transition-all" />
                  </div>
                </div>
              </div>
            </Link>
          </motion.div>
        ))}
      </div>
    </div>
  );
}