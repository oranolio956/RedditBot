"""
Personality Testing and A/B Testing Framework

Comprehensive testing framework for personality system optimization:
- A/B testing different personality approaches
- Performance benchmarking and comparison
- Automated testing of personality algorithms
- User satisfaction measurement
- Personality effectiveness evaluation
"""

import asyncio
import json
import random
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

# Statistical analysis
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu

# Redis for experiment data
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func

# Internal imports
from app.models.personality import PersonalityProfile, UserPersonalityMapping
from app.models.user import User
from app.models.conversation import Message, ConversationSession
from app.services.personality_engine import PersonalityState, ConversationContext
from app.services.personality_matcher import PersonalityMatch, MatchingContext
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class TestType(str, Enum):
    """Types of personality tests."""
    AB_TEST = "ab_test"
    MULTIVARIATE = "multivariate"
    SEQUENTIAL = "sequential"
    REGRESSION = "regression"
    COHORT = "cohort"


class TestStatus(str, Enum):
    """Test status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TestVariant:
    """A variant in a personality test."""
    id: str
    name: str
    description: str
    configuration: Dict[str, Any]
    traffic_allocation: float  # 0.0 to 1.0
    expected_improvement: Optional[float] = None
    minimum_sample_size: int = 100


@dataclass
class TestMetrics:
    """Metrics for test evaluation."""
    primary_metric: str
    secondary_metrics: List[str]
    success_criteria: Dict[str, Any]
    statistical_power: float = 0.8
    significance_level: float = 0.05
    minimum_effect_size: float = 0.05


@dataclass
class TestResult:
    """Result of a personality test."""
    test_id: str
    variant_id: str
    metric_name: str
    sample_size: int
    mean_value: float
    std_deviation: float
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    p_value: float
    effect_size: float
    practical_significance: bool


@dataclass
class PersonalityTest:
    """Complete personality test configuration."""
    id: str
    name: str
    description: str
    test_type: TestType
    variants: List[TestVariant]
    metrics: TestMetrics
    target_population: Dict[str, Any]
    start_date: datetime
    end_date: Optional[datetime]
    status: TestStatus
    sample_size_per_variant: int
    created_by: str
    tags: List[str] = field(default_factory=list)
    notes: str = ""


class PersonalityTestingFramework:
    """
    Advanced A/B testing and experimentation framework for personality systems.
    
    This framework enables rigorous testing of:
    1. Different personality matching algorithms
    2. Adaptation strategies and parameters
    3. Response generation approaches
    4. User interface and experience elements
    5. Performance optimization techniques
    """
    
    def __init__(self, db_session: AsyncSession, redis_client: Redis):
        self.db = db_session
        self.redis = redis_client
        self.settings = get_settings()
        
        # Test management
        self.active_tests = {}  # test_id -> PersonalityTest
        self.user_assignments = {}  # user_id -> {test_id: variant_id}
        self.test_results = defaultdict(list)  # test_id -> [TestResult]
        
        # Statistical analysis tools
        self.statistical_tests = {
            'ttest': self._perform_ttest,
            'mann_whitney': self._perform_mann_whitney,
            'chi_square': self._perform_chi_square,
            'regression': self._perform_regression_analysis
        }
        
        # Performance tracking
        self.experiment_metrics = defaultdict(lambda: defaultdict(list))
        
        logger.info("Personality testing framework initialized")
    
    async def create_ab_test(
        self,
        name: str,
        description: str,
        control_config: Dict[str, Any],
        treatment_config: Dict[str, Any],
        primary_metric: str,
        target_population: Optional[Dict[str, Any]] = None,
        duration_days: int = 14,
        traffic_split: float = 0.5,
        minimum_sample_size: int = 100
    ) -> PersonalityTest:
        """
        Create a new A/B test for personality system optimization.
        
        Args:
            name: Test name
            description: Test description
            control_config: Control variant configuration
            treatment_config: Treatment variant configuration
            primary_metric: Primary metric to optimize
            target_population: User targeting criteria
            duration_days: Test duration
            traffic_split: Traffic allocation for treatment (0-1)
            minimum_sample_size: Minimum users per variant
        """
        try:
            test_id = f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            # Create variants
            control_variant = TestVariant(
                id="control",
                name="Control",
                description="Current baseline implementation",
                configuration=control_config,
                traffic_allocation=1.0 - traffic_split,
                minimum_sample_size=minimum_sample_size
            )
            
            treatment_variant = TestVariant(
                id="treatment",
                name="Treatment",
                description="New implementation to test",
                configuration=treatment_config,
                traffic_allocation=traffic_split,
                minimum_sample_size=minimum_sample_size
            )
            
            # Create test metrics
            metrics = TestMetrics(
                primary_metric=primary_metric,
                secondary_metrics=[
                    "user_satisfaction", "engagement_score", "conversation_length",
                    "response_time", "error_rate"
                ],
                success_criteria={
                    primary_metric: {"improvement": 0.05, "direction": "increase"}
                }
            )
            
            # Create test
            test = PersonalityTest(
                id=test_id,
                name=name,
                description=description,
                test_type=TestType.AB_TEST,
                variants=[control_variant, treatment_variant],
                metrics=metrics,
                target_population=target_population or {},
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=duration_days),
                status=TestStatus.DRAFT,
                sample_size_per_variant=minimum_sample_size,
                created_by="system"
            )
            
            # Store test
            await self._store_test(test)
            
            logger.info(f"Created A/B test: {name} ({test_id})")
            return test
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {e}")
            raise
    
    async def create_multivariate_test(
        self,
        name: str,
        description: str,
        factors: Dict[str, List[Any]],
        primary_metric: str,
        interactions_to_test: Optional[List[Tuple[str, str]]] = None,
        duration_days: int = 21,
        minimum_sample_size: int = 200
    ) -> PersonalityTest:
        """
        Create a multivariate test to test multiple factors simultaneously.
        
        Args:
            name: Test name
            description: Test description
            factors: Dictionary of factor names to their possible values
            primary_metric: Primary metric to optimize
            interactions_to_test: Factor pairs to test for interactions
            duration_days: Test duration
            minimum_sample_size: Minimum users per variant
        """
        try:
            test_id = f"mvt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            # Generate all combinations of factors
            variants = []
            factor_names = list(factors.keys())
            factor_values = list(factors.values())
            
            import itertools
            for i, combination in enumerate(itertools.product(*factor_values)):
                variant_config = dict(zip(factor_names, combination))
                variant_name = "_".join([f"{k}_{v}" for k, v in variant_config.items()])
                
                variant = TestVariant(
                    id=f"variant_{i}",
                    name=variant_name,
                    description=f"Combination: {variant_config}",
                    configuration=variant_config,
                    traffic_allocation=1.0 / len(list(itertools.product(*factor_values))),
                    minimum_sample_size=minimum_sample_size
                )
                variants.append(variant)
            
            # Create metrics
            secondary_metrics = ["user_satisfaction", "engagement_score", "conversation_quality"]
            if interactions_to_test:
                secondary_metrics.extend([f"interaction_{f1}_{f2}" for f1, f2 in interactions_to_test])
            
            metrics = TestMetrics(
                primary_metric=primary_metric,
                secondary_metrics=secondary_metrics,
                success_criteria={
                    primary_metric: {"improvement": 0.03, "direction": "increase"}
                }
            )
            
            # Create test
            test = PersonalityTest(
                id=test_id,
                name=name,
                description=description,
                test_type=TestType.MULTIVARIATE,
                variants=variants,
                metrics=metrics,
                target_population={},
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=duration_days),
                status=TestStatus.DRAFT,
                sample_size_per_variant=minimum_sample_size,
                created_by="system"
            )
            
            await self._store_test(test)
            
            logger.info(f"Created multivariate test: {name} ({test_id}) with {len(variants)} variants")
            return test
            
        except Exception as e:
            logger.error(f"Error creating multivariate test: {e}")
            raise
    
    async def start_test(self, test_id: str) -> bool:
        """Start an experiment."""
        try:
            test = await self._get_test(test_id)
            if not test:
                logger.error(f"Test not found: {test_id}")
                return False
            
            if test.status != TestStatus.DRAFT:
                logger.error(f"Test {test_id} cannot be started (status: {test.status})")
                return False
            
            # Validate test configuration
            validation_errors = await self._validate_test(test)
            if validation_errors:
                logger.error(f"Test validation failed: {validation_errors}")
                return False
            
            # Update status
            test.status = TestStatus.ACTIVE
            test.start_date = datetime.now()
            
            # Store updated test
            await self._store_test(test)
            self.active_tests[test_id] = test
            
            logger.info(f"Started test: {test.name} ({test_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error starting test {test_id}: {e}")
            return False
    
    async def assign_user_to_test(
        self,
        user_id: str,
        test_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Assign user to a test variant.
        
        Returns the assigned variant ID or None if not eligible.
        """
        try:
            test = self.active_tests.get(test_id) or await self._get_test(test_id)
            if not test or test.status != TestStatus.ACTIVE:
                return None
            
            # Check if user already assigned
            if user_id in self.user_assignments and test_id in self.user_assignments[user_id]:
                return self.user_assignments[user_id][test_id]
            
            # Check eligibility
            if not await self._is_user_eligible(user_id, test, context):
                return None
            
            # Assign variant based on traffic allocation
            variant_id = await self._assign_variant(user_id, test)
            
            # Store assignment
            if user_id not in self.user_assignments:
                self.user_assignments[user_id] = {}
            self.user_assignments[user_id][test_id] = variant_id
            
            # Cache assignment
            await self._cache_user_assignment(user_id, test_id, variant_id)
            
            logger.debug(f"Assigned user {user_id} to test {test_id}, variant {variant_id}")
            return variant_id
            
        except Exception as e:
            logger.error(f"Error assigning user to test: {e}")
            return None
    
    async def record_test_event(
        self,
        user_id: str,
        test_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record an event for test analysis."""
        try:
            if user_id not in self.user_assignments or test_id not in self.user_assignments[user_id]:
                return  # User not in test
            
            variant_id = self.user_assignments[user_id][test_id]
            timestamp = timestamp or datetime.now()
            
            event = {
                'user_id': user_id,
                'test_id': test_id,
                'variant_id': variant_id,
                'event_type': event_type,
                'event_data': event_data,
                'timestamp': timestamp.isoformat()
            }
            
            # Store in Redis for real-time analysis
            events_key = f"test_events:{test_id}:{variant_id}"
            await self.redis.lpush(events_key, json.dumps(event))
            await self.redis.expire(events_key, 86400 * 30)  # 30 days
            
            # Update metrics
            await self._update_test_metrics(test_id, variant_id, event_data)
            
        except Exception as e:
            logger.error(f"Error recording test event: {e}")
    
    async def get_test_results(
        self,
        test_id: str,
        metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive test results."""
        try:
            test = await self._get_test(test_id)
            if not test:
                return {'error': f'Test {test_id} not found'}
            
            # Get raw data for each variant
            variant_data = {}
            for variant in test.variants:
                data = await self._get_variant_data(test_id, variant.id)
                variant_data[variant.id] = data
            
            # Calculate statistical results
            results = {}
            metrics_to_analyze = [test.metrics.primary_metric]
            if not metric_name:
                metrics_to_analyze.extend(test.metrics.secondary_metrics)
            elif metric_name != test.metrics.primary_metric:
                metrics_to_analyze = [metric_name]
            
            for metric in metrics_to_analyze:
                metric_results = await self._analyze_metric(test, variant_data, metric)
                results[metric] = metric_results
            
            # Overall test summary
            summary = await self._generate_test_summary(test, results)
            
            return {
                'test_id': test_id,
                'test_name': test.name,
                'status': test.status.value,
                'start_date': test.start_date.isoformat(),
                'end_date': test.end_date.isoformat() if test.end_date else None,
                'summary': summary,
                'results': results,
                'variant_data': {
                    variant_id: {
                        'sample_size': len(data),
                        'configuration': next(v.configuration for v in test.variants if v.id == variant_id)
                    }
                    for variant_id, data in variant_data.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting test results: {e}")
            return {'error': str(e)}
    
    async def stop_test(
        self,
        test_id: str,
        reason: str = "Completed naturally"
    ) -> bool:
        """Stop a running test."""
        try:
            test = await self._get_test(test_id)
            if not test:
                return False
            
            test.status = TestStatus.COMPLETED
            test.end_date = datetime.now()
            test.notes += f"\nStopped: {reason} at {datetime.now().isoformat()}"
            
            await self._store_test(test)
            
            if test_id in self.active_tests:
                del self.active_tests[test_id]
            
            # Generate final results
            final_results = await self.get_test_results(test_id)
            await self._store_final_results(test_id, final_results)
            
            logger.info(f"Stopped test: {test.name} ({test_id}). Reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping test {test_id}: {e}")
            return False
    
    async def analyze_personality_algorithm_performance(
        self,
        algorithm_names: List[str],
        metric_name: str = "user_satisfaction",
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Analyze and compare different personality algorithm performance."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            algorithm_performance = {}
            
            for algorithm in algorithm_names:
                # Get performance data for algorithm
                data = await self._get_algorithm_performance_data(
                    algorithm, metric_name, start_date, end_date
                )
                
                if data:
                    algorithm_performance[algorithm] = {
                        'sample_size': len(data),
                        'mean': np.mean(data),
                        'std': np.std(data),
                        'median': np.median(data),
                        'min': np.min(data),
                        'max': np.max(data),
                        'percentiles': {
                            '25th': np.percentile(data, 25),
                            '75th': np.percentile(data, 75),
                            '95th': np.percentile(data, 95)
                        }
                    }
            
            # Perform statistical comparisons
            comparisons = {}
            algorithms = list(algorithm_performance.keys())
            
            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    algo1, algo2 = algorithms[i], algorithms[j]
                    
                    data1 = await self._get_algorithm_performance_data(
                        algo1, metric_name, start_date, end_date
                    )
                    data2 = await self._get_algorithm_performance_data(
                        algo2, metric_name, start_date, end_date
                    )
                    
                    if data1 and data2:
                        comparison = await self._compare_algorithms(data1, data2, algo1, algo2)
                        comparisons[f"{algo1}_vs_{algo2}"] = comparison
            
            return {
                'analysis_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': days_back
                },
                'metric': metric_name,
                'algorithm_performance': algorithm_performance,
                'statistical_comparisons': comparisons,
                'recommendations': await self._generate_algorithm_recommendations(
                    algorithm_performance, comparisons
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing algorithm performance: {e}")
            return {'error': str(e)}
    
    async def run_personality_effectiveness_benchmark(
        self,
        personality_profiles: List[str],
        test_scenarios: List[Dict[str, Any]],
        duration_hours: int = 24
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark of personality effectiveness."""
        try:
            benchmark_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting personality effectiveness benchmark: {benchmark_id}")
            
            benchmark_results = {
                'benchmark_id': benchmark_id,
                'start_time': datetime.now().isoformat(),
                'profiles_tested': personality_profiles,
                'scenarios': test_scenarios,
                'results': {}
            }
            
            # Run tests for each personality profile
            for profile_id in personality_profiles:
                profile_results = []
                
                for scenario in test_scenarios:
                    scenario_result = await self._run_scenario_test(
                        profile_id, scenario, duration_hours
                    )
                    profile_results.append(scenario_result)
                
                # Calculate aggregate metrics for profile
                benchmark_results['results'][profile_id] = {
                    'scenario_results': profile_results,
                    'aggregate_metrics': self._calculate_aggregate_metrics(profile_results),
                    'performance_score': self._calculate_performance_score(profile_results)
                }
            
            # Generate recommendations
            benchmark_results['recommendations'] = await self._generate_benchmark_recommendations(
                benchmark_results['results']
            )
            
            benchmark_results['end_time'] = datetime.now().isoformat()
            benchmark_results['duration_hours'] = duration_hours
            
            # Store benchmark results
            await self._store_benchmark_results(benchmark_id, benchmark_results)
            
            logger.info(f"Completed personality effectiveness benchmark: {benchmark_id}")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Error running personality effectiveness benchmark: {e}")
            return {'error': str(e)}
    
    # Statistical analysis methods
    
    async def _perform_ttest(
        self,
        data1: List[float],
        data2: List[float],
        variant1_name: str,
        variant2_name: str
    ) -> Dict[str, Any]:
        """Perform t-test comparison."""
        try:
            # Check assumptions
            if len(data1) < 30 or len(data2) < 30:
                logger.warning("Small sample sizes for t-test")
            
            # Perform Welch's t-test (doesn't assume equal variances)
            statistic, p_value = ttest_ind(data1, data2, equal_var=False)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
            
            # Confidence interval for difference
            se_diff = np.sqrt(np.var(data1, ddof=1) / len(data1) + np.var(data2, ddof=1) / len(data2))
            diff = np.mean(data1) - np.mean(data2)
            ci_lower = diff - 1.96 * se_diff
            ci_upper = diff + 1.96 * se_diff
            
            return {
                'test_type': 'ttest',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'effect_size': float(cohens_d),
                'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
                'mean_difference': float(diff),
                'confidence_interval': [float(ci_lower), float(ci_upper)],
                'sample_sizes': {'group1': len(data1), 'group2': len(data2)},
                'group_means': {'group1': float(np.mean(data1)), 'group2': float(np.mean(data2))}
            }
            
        except Exception as e:
            logger.error(f"Error performing t-test: {e}")
            return {'error': str(e)}
    
    async def _perform_mann_whitney(
        self,
        data1: List[float],
        data2: List[float],
        variant1_name: str,
        variant2_name: str
    ) -> Dict[str, Any]:
        """Perform Mann-Whitney U test (non-parametric)."""
        try:
            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            
            # Effect size for Mann-Whitney (r = Z / sqrt(N))
            n1, n2 = len(data1), len(data2)
            z_score = stats.norm.ppf(1 - p_value / 2)  # Two-tailed
            r_effect_size = z_score / np.sqrt(n1 + n2)
            
            return {
                'test_type': 'mann_whitney',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'effect_size_r': float(r_effect_size),
                'effect_size_interpretation': self._interpret_r_effect_size(r_effect_size),
                'sample_sizes': {'group1': n1, 'group2': n2},
                'group_medians': {'group1': float(np.median(data1)), 'group2': float(np.median(data2))}
            }
            
        except Exception as e:
            logger.error(f"Error performing Mann-Whitney test: {e}")
            return {'error': str(e)}
    
    async def _perform_chi_square(
        self,
        contingency_table: List[List[int]],
        variant_names: List[str]
    ) -> Dict[str, Any]:
        """Perform chi-square test of independence."""
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Cramér's V (effect size for chi-square)
            n = np.sum(contingency_table)
            cramers_v = np.sqrt(chi2 / (n * (min(len(contingency_table), len(contingency_table[0])) - 1)))
            
            return {
                'test_type': 'chi_square',
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < 0.05,
                'effect_size_cramers_v': float(cramers_v),
                'effect_size_interpretation': self._interpret_cramers_v(cramers_v),
                'observed_frequencies': contingency_table,
                'expected_frequencies': expected.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error performing chi-square test: {e}")
            return {'error': str(e)}
    
    async def _perform_regression_analysis(
        self,
        data: List[Dict[str, float]],
        dependent_var: str,
        independent_vars: List[str]
    ) -> Dict[str, Any]:
        """Perform regression analysis."""
        try:
            import pandas as pd
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score, mean_squared_error
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            X = df[independent_vars]
            y = df[dependent_var]
            
            # Fit regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predictions and metrics
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            # Feature importance
            feature_importance = dict(zip(independent_vars, model.coef_))
            
            return {
                'test_type': 'regression',
                'r_squared': float(r2),
                'rmse': float(rmse),
                'intercept': float(model.intercept_),
                'coefficients': {var: float(coef) for var, coef in feature_importance.items()},
                'feature_importance': feature_importance,
                'sample_size': len(data)
            }
            
        except Exception as e:
            logger.error(f"Error performing regression analysis: {e}")
            return {'error': str(e)}
    
    # Helper methods for effect size interpretation
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_r_effect_size(self, r: float) -> str:
        """Interpret r effect size."""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramér's V effect size."""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
    
    # Additional helper methods would be implemented here...
    # For brevity, including key method signatures:
    
    async def _store_test(self, test: PersonalityTest) -> None:
        """Store test configuration."""
        pass
    
    async def _get_test(self, test_id: str) -> Optional[PersonalityTest]:
        """Retrieve test configuration."""
        pass
    
    async def _validate_test(self, test: PersonalityTest) -> List[str]:
        """Validate test configuration."""
        pass
    
    async def _is_user_eligible(self, user_id: str, test: PersonalityTest, context: Optional[Dict]) -> bool:
        """Check if user is eligible for test."""
        pass
    
    async def _assign_variant(self, user_id: str, test: PersonalityTest) -> str:
        """Assign user to test variant."""
        pass
    
    async def _cache_user_assignment(self, user_id: str, test_id: str, variant_id: str) -> None:
        """Cache user assignment."""
        pass
    
    async def _update_test_metrics(self, test_id: str, variant_id: str, event_data: Dict) -> None:
        """Update test metrics."""
        pass
    
    async def _get_variant_data(self, test_id: str, variant_id: str) -> List[Dict[str, Any]]:
        """Get data for test variant."""
        pass
    
    async def _analyze_metric(self, test: PersonalityTest, variant_data: Dict, metric: str) -> Dict:
        """Analyze specific metric across variants."""
        pass
    
    async def _generate_test_summary(self, test: PersonalityTest, results: Dict) -> Dict:
        """Generate test summary."""
        pass


# Export main classes
__all__ = [
    'PersonalityTestingFramework', 'PersonalityTest', 'TestVariant', 
    'TestMetrics', 'TestResult', 'TestType', 'TestStatus'
]