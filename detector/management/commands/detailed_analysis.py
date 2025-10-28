"""
Detailed analysis command to understand failure patterns and success factors
"""

from django.core.management.base import BaseCommand
from detector.models import ImageUpload
from django.utils import timezone
from datetime import timedelta
import json

class Command(BaseCommand):
    help = 'Detailed analysis of detection failures and successes'

    def add_arguments(self, parser):
        parser.add_argument(
            '--limit',
            type=int,
            default=50,
            help='Number of recent feedback samples to analyze',
        )

    def handle(self, *args, **options):
        limit = options['limit']
        
        # Get recent uploads with feedback
        uploads = ImageUpload.objects.filter(
            user_feedback__in=['correct', 'incorrect']
        ).order_by('-uploaded_at')[:limit]
        
        self.stdout.write("=" * 80)
        self.stdout.write("DETAILED FAILURE PATTERN ANALYSIS")
        self.stdout.write("=" * 80)
        
        # Analyze success factors
        success_factors = self.analyze_success_factors(uploads)
        failure_patterns = self.analyze_failure_patterns(uploads)
        
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("SUCCESS FACTORS - What's Always Working")
        self.stdout.write("=" * 80)
        self.stdout.write(success_factors)
        
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("FAILURE PATTERNS - Why We Fail")
        self.stdout.write("=" * 80)
        self.stdout.write(failure_patterns)
        
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("RECOMMENDATIONS")
        self.stdout.write("=" * 80)
        
        recommendations = self.generate_recommendations(uploads)
        self.stdout.write(recommendations)
        
        self.stdout.write("\n" + "=" * 80)
    
    def analyze_success_factors(self, uploads):
        """Identify what consistently works well"""
        factors = {
            'method_2_indicators': {},
            'method_3_indicators': {},
            'confidence_ranges': {'correct': [], 'incorrect': []},
            'agreement_patterns': {'unanimous': 0, 'majority': 0, 'split': 0}
        }
        
        for upload in uploads:
            # Determine actual label
            actual_is_ai = upload.is_ai_generated if upload.user_feedback == 'correct' else not upload.is_ai_generated
            
            if upload.analysis_details and 'method_comparison' in upload.analysis_details:
                comp = upload.analysis_details['method_comparison']
                
                # Track indicator accuracy for Method 2
                if 'method_2' in comp and upload.user_feedback == 'correct':
                    indicators = comp['method_2'].get('indicators', [])
                    for indicator in indicators:
                        if indicator not in factors['method_2_indicators']:
                            factors['method_2_indicators'][indicator] = {'correct': 0, 'total': 0}
                        factors['method_2_indicators'][indicator]['correct'] += 1
                        factors['method_2_indicators'][indicator]['total'] += 1
                
                # Track confidence ranges
                if 'method_2' in comp:
                    conf = comp['method_2'].get('confidence', 0)
                    factors['confidence_ranges']['correct' if upload.user_feedback == 'correct' else 'incorrect'].append(conf)
                
                # Track agreement
                if 'agreement' in upload.analysis_details:
                    agreement = upload.analysis_details['agreement']
                    if 'unanimous' in str(agreement).lower():
                        factors['agreement_patterns']['unanimous'] += 1
                    elif 'majority' in str(agreement).lower():
                        factors['agreement_patterns']['majority'] += 1
                    else:
                        factors['agreement_patterns']['split'] += 1
        
        # Build output
        output = []
        
        # Most reliable indicators
        output.append("\n⭐ MOST RELIABLE INDICATORS (Method 2):")
        sorted_indicators = sorted(
            factors['method_2_indicators'].items(),
            key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0,
            reverse=True
        )
        for indicator, stats in sorted_indicators[:10]:
            accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            if accuracy >= 70 and stats['total'] >= 5:
                output.append(f"  ✓ {indicator[:60]}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
        
        # Confidence analysis
        if factors['confidence_ranges']['correct']:
            avg_correct_conf = sum(factors['confidence_ranges']['correct']) / len(factors['confidence_ranges']['correct'])
            output.append(f"\n📊 Confidence Analysis:")
            output.append(f"  Average confidence on CORRECT detections: {avg_correct_conf*100:.1f}%")
            if factors['confidence_ranges']['incorrect']:
                avg_incorrect_conf = sum(factors['confidence_ranges']['incorrect']) / len(factors['confidence_ranges']['incorrect'])
                output.append(f"  Average confidence on INCORRECT detections: {avg_incorrect_conf*100:.1f}%")
        
        # Agreement analysis
        total = sum(factors['agreement_patterns'].values())
        if total > 0:
            output.append(f"\n🤝 Method Agreement Analysis:")
            output.append(f"  Unanimous agreement: {factors['agreement_patterns']['unanimous']}/{total} ({factors['agreement_patterns']['unanimous']/total*100:.1f}%)")
            output.append(f"  Majority agreement: {factors['agreement_patterns']['majority']}/{total} ({factors['agreement_patterns']['majority']/total*100:.1f}%)")
        
        return "\n".join(output)
    
    def analyze_failure_patterns(self, uploads):
        """Identify why detections fail"""
        false_positives = []  # Real detected as AI
        false_negatives = []  # AI detected as Real
        
        method_errors = {
            'method_1': {'fp': [], 'fn': []},
            'method_2': {'fp': [], 'fn': []},
            'method_3': {'fp': [], 'fn': []}
        }
        
        for upload in uploads:
            actual_is_ai = upload.is_ai_generated if upload.user_feedback == 'correct' else not upload.is_ai_generated
            
            if upload.user_feedback == 'incorrect':
                if actual_is_ai:
                    false_negatives.append(upload)
                else:
                    false_positives.append(upload)
            
            # Method-specific errors
            if upload.analysis_details and 'method_comparison' in upload.analysis_details:
                comp = upload.analysis_details['method_comparison']
                
                for method_key in ['method_1', 'method_2', 'method_3']:
                    if method_key in comp:
                        method_pred = comp[method_key].get('is_ai_generated', False)
                        if method_pred != actual_is_ai:
                            conf = comp[method_key].get('confidence', 0)
                            if actual_is_ai:
                                method_errors[method_key]['fn'].append(conf)
                            else:
                                method_errors[method_key]['fp'].append(conf)
        
        output = []
        
        output.append(f"\n❌ Overall Error Breakdown:")
        output.append(f"  Total Errors: {len(false_positives) + len(false_negatives)}/{len(uploads)}")
        output.append(f"  False Positives (Real→AI): {len(false_positives)} ({len(false_positives)/len(uploads)*100:.1f}%)")
        output.append(f"  False Negatives (AI→Real): {len(false_negatives)} ({len(false_negatives)/len(uploads)*100:.1f}%)")
        
        output.append(f"\n📉 Method-Specific Error Patterns:")
        
        for method_key, method_name in [('method_1', 'Method 1'), ('method_2', 'Method 2'), ('method_3', 'Method 3')]:
            fp = method_errors[method_key]['fp']
            fn = method_errors[method_key]['fn']
            
            output.append(f"\n  {method_name}:")
            output.append(f"    False Positives: {len(fp)}")
            if fp:
                avg_conf = sum(fp) / len(fp)
                output.append(f"      Average confidence on errors: {avg_conf*100:.1f}%")
            
            output.append(f"    False Negatives: {len(fn)}")
            if fn:
                avg_conf = sum(fn) / len(fn)
                output.append(f"      Average confidence on errors: {avg_conf*100:.1f}%")
            
            total_errors = len(fp) + len(fn)
            if total_errors > 0:
                output.append(f"    Error Type: {'More false negatives' if len(fn) > len(fp) else 'More false positives'}")
        
        return "\n".join(output)
    
    def generate_recommendations(self, uploads):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Analyze what's working
        method_correct = {'method_1': 0, 'method_2': 0, 'method_3': 0}
        method_total = {'method_1': 0, 'method_2': 0, 'method_3': 0}
        
        false_positives = []
        false_negatives = []
        
        for upload in uploads:
            actual_is_ai = upload.is_ai_generated if upload.user_feedback == 'correct' else not upload.is_ai_generated
            
            if upload.user_feedback == 'incorrect':
                if actual_is_ai:
                    false_negatives.append(upload)
                else:
                    false_positives.append(upload)
            
            if upload.analysis_details and 'method_comparison' in upload.analysis_details:
                comp = upload.analysis_details['method_comparison']
                
                for method_key in ['method_1', 'method_2', 'method_3']:
                    if method_key in comp:
                        method_total[method_key] += 1
                        method_pred = comp[method_key].get('is_ai_generated', False)
                        if method_pred == actual_is_ai:
                            method_correct[method_key] += 1
        
        # Calculate accuracies
        accuracies = {}
        for method_key in ['method_1', 'method_2', 'method_3']:
            if method_total[method_key] > 0:
                accuracies[method_key] = method_correct[method_key] / method_total[method_key]
            else:
                accuracies[method_key] = 0
        
        recommendations.append("\n🎯 KEY RECOMMENDATIONS:\n")
        
        # Method 2 recommendations
        if accuracies.get('method_2', 0) >= 0.70:
            recommendations.append("1. ✅ Method 2 (Statistical) is your STRONGEST method - prioritize it!")
            recommendations.append("   → Current accuracy: {:.1f}%".format(accuracies['method_2']*100))
            recommendations.append("   → Action: Increase Method 2 weight to 75-85%")
        else:
            recommendations.append("1. ⚠️ Method 2 needs improvement")
            recommendations.append("   → Current accuracy: {:.1f}%".format(accuracies.get('method_2', 0)*100))
            recommendations.append("   → Action: Fine-tune statistical thresholds")
        
        # False negative problem (missing AI images)
        fn_ratio = len(false_negatives) / (len(false_negatives) + len(false_positives)) if (false_negatives or false_positives) else 0
        if fn_ratio > 0.6:
            recommendations.append("\n2. ❌ CRITICAL ISSUE: Missing AI-generated images (False Negatives)")
            recommendations.append("   → {:.1f}% of errors are false negatives".format(fn_ratio*100))
            recommendations.append("   → Action: Lower detection thresholds, especially for Method 2")
            recommendations.append("   → Action: Increase sensitivity to AI indicators")
        elif len(false_positives) > len(false_negatives):
            recommendations.append("\n2. ⚠️ ISSUE: False positives (Real images flagged as AI)")
            recommendations.append("   → Action: Increase detection thresholds slightly")
            recommendations.append("   → Action: Require stronger AI indicators before flagging")
        
        # Method 1 recommendations
        if accuracies.get('method_1', 0) < 0.50:
            recommendations.append("\n3. ❌ Method 1 (Deep Learning) is underperforming")
            recommendations.append("   → Current accuracy: {:.1f}%".format(accuracies.get('method_1', 0)*100))
            recommendations.append("   → Action: Reduce Method 1 weight to 5-10%")
            recommendations.append("   → Future: Consider fine-tuning models on AI detection dataset")
        
        # Method 3 recommendations
        if accuracies.get('method_3', 0) < 0.50:
            recommendations.append("\n4. ❌ Method 3 (Forensics) is underperforming")
            recommendations.append("   → Current accuracy: {:.1f}%".format(accuracies.get('method_3', 0)*100))
            recommendations.append("   → Action: Reduce Method 3 weight to 5-10%")
            recommendations.append("   → Action: Adjust forensics thresholds to catch more AI images")
        
        # Threshold recommendations
        recommendations.append("\n5. 🔧 Threshold Adjustments:")
        recommendations.append("   → If missing AI images: Lower Method 2 threshold from 0.33 to 0.30")
        recommendations.append("   → If flagging too many real images: Raise Method 2 threshold to 0.35")
        recommendations.append("   → Boost high-accuracy factors: Increase weights for indicators with 70%+ accuracy")
        
        # Weight distribution recommendation
        best_method = max(accuracies.items(), key=lambda x: x[1])
        recommendations.append(f"\n6. ⚖️ Recommended Weight Distribution:")
        recommendations.append(f"   → Method 2 (Best at {best_method[1]*100:.1f}%): 75-85%")
        recommendations.append(f"   → Method 1: 10-15%")
        recommendations.append(f"   → Method 3: 5-10%")
        
        return "\n".join(recommendations)

