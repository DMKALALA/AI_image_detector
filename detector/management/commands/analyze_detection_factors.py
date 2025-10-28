from django.core.management.base import BaseCommand
from detector.models import ImageUpload
from django.db.models import Q
import json
from collections import defaultdict

class Command(BaseCommand):
    help = 'Analyze which specific factors in each method contribute to correct detections'

    def add_arguments(self, parser):
        parser.add_argument(
            '--limit',
            type=int,
            default=60,
            help='Number of recent uploads to analyze (default: 60)',
        )

    def handle(self, *args, **options):
        limit = options['limit']
        
        self.stdout.write(self.style.SUCCESS('=' * 70))
        self.stdout.write(self.style.SUCCESS('DETAILED FACTOR-LEVEL ANALYSIS'))
        self.stdout.write(self.style.SUCCESS('=' * 70))
        
        # Get recent uploads with feedback
        recent_uploads = ImageUpload.objects.filter(
            ~Q(user_feedback='')
        ).order_by('-uploaded_at')[:limit]
        
        if not recent_uploads:
            self.stdout.write(self.style.WARNING('No uploads with feedback found.'))
            return
        
        self.stdout.write(f'\nAnalyzing {len(recent_uploads)} recent uploads...\n')
        
        # Factor analysis for each method
        method_1_factors = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'total': 0})
        method_2_factors = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'total': 0})
        method_3_factors = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'total': 0})
        
        # Method accuracy tracking
        method_accuracy = {
            'method_1': {'correct': 0, 'total': 0, 'never_accurate': True},
            'method_2': {'correct': 0, 'total': 0, 'never_accurate': True},
            'method_3': {'correct': 0, 'total': 0, 'never_accurate': True}
        }
        
        for upload in recent_uploads:
            # Get ground truth from feedback
            if upload.user_feedback == 'correct':
                ground_truth = upload.is_ai_generated
            elif upload.user_feedback == 'incorrect':
                ground_truth = not upload.is_ai_generated
            else:
                continue
            
            analysis_details = upload.analysis_details or {}
            method_comparison = analysis_details.get('method_comparison', {})
            
            if not method_comparison:
                continue
            
            # Analyze Method 1
            m1 = method_comparison.get('method_1', {})
            if m1.get('available'):
                m1_correct = m1.get('is_ai_generated') == ground_truth
                method_accuracy['method_1']['total'] += 1
                if m1_correct:
                    method_accuracy['method_1']['correct'] += 1
                    method_accuracy['method_1']['never_accurate'] = False
                
                # Analyze indicators (factors)
                indicators = m1.get('indicators', [])
                for indicator in indicators:
                    # Extract factor type from indicator
                    factor = self._extract_factor_from_indicator(indicator, 'method_1')
                    if factor:
                        method_1_factors[factor]['total'] += 1
                        if m1_correct:
                            method_1_factors[factor]['correct'] += 1
                        else:
                            method_1_factors[factor]['incorrect'] += 1
            
            # Analyze Method 2
            m2 = method_comparison.get('method_2', {})
            m2_correct = m2.get('is_ai_generated') == ground_truth
            method_accuracy['method_2']['total'] += 1
            if m2_correct:
                method_accuracy['method_2']['correct'] += 1
                method_accuracy['method_2']['never_accurate'] = False
            
            # Extract factors from analysis_details
            all_methods = analysis_details.get('all_methods', {})
            m2_result = all_methods.get('method_2', {})
            factors_detected = m2_result.get('factors_detected', [])
            
            # Track each factor
            for factor in factors_detected:
                method_2_factors[factor]['total'] += 1
                if m2_correct:
                    method_2_factors[factor]['correct'] += 1
                else:
                    method_2_factors[factor]['incorrect'] += 1
            
            # Also track when no factors detected
            if not factors_detected:
                method_2_factors['no_factors']['total'] += 1
                if m2_correct:
                    method_2_factors['no_factors']['correct'] += 1
                else:
                    method_2_factors['no_factors']['incorrect'] += 1
            
            # Analyze Method 3
            m3 = method_comparison.get('method_3', {})
            m3_correct = m3.get('is_ai_generated') == ground_truth
            method_accuracy['method_3']['total'] += 1
            if m3_correct:
                method_accuracy['method_3']['correct'] += 1
                method_accuracy['method_3']['never_accurate'] = False
            
            # Extract factors from Method 3
            m3_result = all_methods.get('method_3', {})
            m3_factors = m3_result.get('factors_detected', [])
            
            for factor in m3_factors:
                method_3_factors[factor]['total'] += 1
                if m3_correct:
                    method_3_factors[factor]['correct'] += 1
                else:
                    method_3_factors[factor]['incorrect'] += 1
            
            if not m3_factors:
                method_3_factors['no_factors']['total'] += 1
                if m3_correct:
                    method_3_factors['no_factors']['correct'] += 1
                else:
                    method_3_factors['no_factors']['incorrect'] += 1
        
        # Print Method Accuracy Summary
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.SUCCESS('METHOD ACCURACY SUMMARY'))
        self.stdout.write('=' * 70)
        
        for method_key, stats in method_accuracy.items():
            method_name = {
                'method_1': 'Method 1: Deep Learning',
                'method_2': 'Method 2: Statistical',
                'method_3': 'Method 3: Metadata'
            }.get(method_key, method_key)
            
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                status = "❌ NEVER ACCURATE" if stats['never_accurate'] else f"✅ {accuracy:.1f}%"
                self.stdout.write(f"{method_name}: {status} ({stats['correct']}/{stats['total']})")
            else:
                self.stdout.write(f"{method_name}: No data")
        
        # Print Method 2 Factor Analysis
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.SUCCESS('METHOD 2 FACTOR ANALYSIS (Best Method)'))
        self.stdout.write('=' * 70)
        
        high_accuracy_factors = []
        
        for factor, stats in sorted(method_2_factors.items(), key=lambda x: x[1]['total'], reverse=True):
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                if accuracy >= 65:
                    high_accuracy_factors.append((factor, accuracy, stats))
                    self.stdout.write(self.style.SUCCESS(
                        f"\n✓ {factor}: {accuracy:.1f}% accuracy ({stats['correct']}/{stats['total']}) - HIGH ACCURACY"
                    ))
                else:
                    self.stdout.write(
                        f"\n  {factor}: {accuracy:.1f}% accuracy ({stats['correct']}/{stats['total']})"
                    )
        
        # Print Method 3 Factor Analysis
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.SUCCESS('METHOD 3 FACTOR ANALYSIS'))
        self.stdout.write('=' * 70)
        
        method_3_high_accuracy = []
        
        for factor, stats in sorted(method_3_factors.items(), key=lambda x: x[1]['total'], reverse=True):
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                if accuracy >= 65:
                    method_3_high_accuracy.append((factor, accuracy, stats))
                    self.stdout.write(self.style.SUCCESS(
                        f"\n✓ {factor}: {accuracy:.1f}% accuracy ({stats['correct']}/{stats['total']}) - HIGH ACCURACY"
                    ))
                else:
                    self.stdout.write(
                        f"\n  {factor}: {accuracy:.1f}% accuracy ({stats['correct']}/{stats['total']})"
                    )
        
        # Recommendations
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.SUCCESS('RECOMMENDATIONS'))
        self.stdout.write('=' * 70)
        
        # Check if methods should be replaced
        m1_acc = (method_accuracy['method_1']['correct'] / method_accuracy['method_1']['total'] * 100) if method_accuracy['method_1']['total'] > 0 else 0
        m2_acc = (method_accuracy['method_2']['correct'] / method_accuracy['method_2']['total'] * 100) if method_accuracy['method_2']['total'] > 0 else 0
        m3_acc = (method_accuracy['method_3']['correct'] / method_accuracy['method_3']['total'] * 100) if method_accuracy['method_3']['total'] > 0 else 0
        
        self.stdout.write(f"\n1. Method Performance:")
        self.stdout.write(f"   Method 1: {m1_acc:.1f}% - {'REPLACE RECOMMENDED' if m1_acc < 40 else 'Keep but reduce weight'}")
        self.stdout.write(f"   Method 2: {m2_acc:.1f}% - {'Good performance' if m2_acc >= 60 else 'Needs improvement'}")
        self.stdout.write(f"   Method 3: {m3_acc:.1f}% - {'REPLACE RECOMMENDED' if m3_acc < 40 else 'Keep but reduce weight'}")
        
        # Suggest weight changes based on accuracy
        total_acc = m1_acc + m2_acc + m3_acc
        if total_acc > 0:
            self.stdout.write(f"\n2. Suggested Weight Distribution (based on accuracy):")
            if m1_acc >= 0:
                w1 = m1_acc / total_acc
                self.stdout.write(f"   Method 1: {w1:.2f} ({(w1*100):.1f}%)")
            if m2_acc >= 0:
                w2 = m2_acc / total_acc
                self.stdout.write(f"   Method 2: {w2:.2f} ({(w2*100):.1f}%)")
            if m3_acc >= 0:
                w3 = m3_acc / total_acc
                self.stdout.write(f"   Method 3: {w3:.2f} ({(w3*100):.1f}%)")
        
        # Factor recommendations
        if high_accuracy_factors:
            self.stdout.write(f"\n3. High-Accuracy Factors in Method 2 (65%+):")
            for factor, accuracy, stats in high_accuracy_factors:
                self.stdout.write(f"   ✓ {factor}: {accuracy:.1f}% - BOOST WEIGHT")
        
        if method_3_high_accuracy:
            self.stdout.write(f"\n4. High-Accuracy Factors in Method 3 (65%+):")
            for factor, accuracy, stats in method_3_high_accuracy:
                self.stdout.write(f"   ✓ {factor}: {accuracy:.1f}% - BOOST WEIGHT")
        
        self.stdout.write('\n' + '=' * 70 + '\n')
    
    def _extract_factor_from_indicator(self, indicator: str, method: str) -> str:
        """Extract factor name from indicator text"""
        indicator_lower = indicator.lower()
        
        if method == 'method_1':
            if 'neural network' in indicator_lower or 'model' in indicator_lower:
                return 'neural_network_prediction'
            elif 'confidence' in indicator_lower:
                return 'model_confidence'
        
        return None

