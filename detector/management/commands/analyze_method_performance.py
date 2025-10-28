from django.core.management.base import BaseCommand
from detector.models import ImageUpload
from django.db.models import Q
import json

class Command(BaseCommand):
    help = 'Analyze recent detection results and method performance to identify improvements'

    def add_arguments(self, parser):
        parser.add_argument(
            '--limit',
            type=int,
            default=20,
            help='Number of recent uploads to analyze (default: 20)',
        )

    def handle(self, *args, **options):
        limit = options['limit']
        
        self.stdout.write(self.style.SUCCESS('=' * 70))
        self.stdout.write(self.style.SUCCESS('METHOD PERFORMANCE ANALYSIS'))
        self.stdout.write(self.style.SUCCESS('=' * 70))
        
        # Get recent uploads with feedback
        recent_uploads = ImageUpload.objects.filter(
            ~Q(user_feedback='')
        ).order_by('-uploaded_at')[:limit]
        
        if not recent_uploads:
            self.stdout.write(self.style.WARNING('No uploads with feedback found.'))
            return
        
        self.stdout.write(f'\nAnalyzing {len(recent_uploads)} recent uploads with feedback...\n')
        
        # Analyze method performance
        method_stats = {
            'method_1': {'correct': 0, 'incorrect': 0, 'total': 0, 'confidences': []},
            'method_2': {'correct': 0, 'incorrect': 0, 'total': 0, 'confidences': []},
            'method_3': {'correct': 0, 'incorrect': 0, 'total': 0, 'confidences': []},
            'final': {'correct': 0, 'incorrect': 0, 'total': 0, 'confidences': []}
        }
        
        agreement_analysis = {
            'unanimous_agreement': 0,
            'majority_agreement': 0,
            'disagreement': 0,
            'method_1_wins': 0,
            'method_2_wins': 0,
            'method_3_wins': 0
        }
        
        issues = []
        
        for upload in recent_uploads:
            # Get ground truth from feedback
            if upload.user_feedback == 'correct':
                ground_truth = upload.is_ai_generated  # If correct, our detection matches truth
            elif upload.user_feedback == 'incorrect':
                ground_truth = not upload.is_ai_generated  # If incorrect, flip it
            else:
                continue  # Skip unsure
            
            # Analyze each method
            analysis_details = upload.analysis_details or {}
            method_comparison = analysis_details.get('method_comparison', {})
            
            if not method_comparison or not method_comparison.get('method_1'):
                continue  # Skip if no method comparison data
            
            # Method 1 (Deep Learning)
            m1 = method_comparison.get('method_1', {})
            m1_correct = m1.get('is_ai_generated') == ground_truth
            if m1.get('available'):
                method_stats['method_1']['total'] += 1
                if m1_correct:
                    method_stats['method_1']['correct'] += 1
                else:
                    method_stats['method_1']['incorrect'] += 1
                    issues.append({
                        'image_id': upload.pk,
                        'method': 'Method 1 (Deep Learning)',
                        'predicted': 'AI' if m1.get('is_ai_generated') else 'Real',
                        'actual': 'AI' if ground_truth else 'Real',
                        'confidence': m1.get('confidence', 0)
                    })
                method_stats['method_1']['confidences'].append(m1.get('confidence', 0))
            
            # Method 2 (Statistical)
            m2 = method_comparison.get('method_2', {})
            m2_correct = m2.get('is_ai_generated') == ground_truth
            method_stats['method_2']['total'] += 1
            if m2_correct:
                method_stats['method_2']['correct'] += 1
            else:
                method_stats['method_2']['incorrect'] += 1
                issues.append({
                    'image_id': upload.pk,
                    'method': 'Method 2 (Statistical)',
                    'predicted': 'AI' if m2.get('is_ai_generated') else 'Real',
                    'actual': 'AI' if ground_truth else 'Real',
                    'confidence': m2.get('confidence', 0)
                })
            method_stats['method_2']['confidences'].append(m2.get('confidence', 0))
            
            # Method 3 (Metadata)
            m3 = method_comparison.get('method_3', {})
            m3_correct = m3.get('is_ai_generated') == ground_truth
            method_stats['method_3']['total'] += 1
            if m3_correct:
                method_stats['method_3']['correct'] += 1
            else:
                method_stats['method_3']['incorrect'] += 1
                issues.append({
                    'image_id': upload.pk,
                    'method': 'Method 3 (Metadata)',
                    'predicted': 'AI' if m3.get('is_ai_generated') else 'Real',
                    'actual': 'AI' if ground_truth else 'Real',
                    'confidence': m3.get('confidence', 0)
                })
            method_stats['method_3']['confidences'].append(m3.get('confidence', 0))
            
            # Final result
            final_correct = upload.is_ai_generated == ground_truth
            method_stats['final']['total'] += 1
            if final_correct:
                method_stats['final']['correct'] += 1
            else:
                method_stats['final']['incorrect'] += 1
            method_stats['final']['confidences'].append(upload.confidence_score)
            
            # Agreement analysis
            decisions = [
                m1.get('is_ai_generated') if m1.get('available') else None,
                m2.get('is_ai_generated'),
                m3.get('is_ai_generated')
            ]
            decisions = [d for d in decisions if d is not None]
            
            if len(set(decisions)) == 1:
                agreement_analysis['unanimous_agreement'] += 1
            elif len(set(decisions)) == 2:
                agreement_analysis['majority_agreement'] += 1
            else:
                agreement_analysis['disagreement'] += 1
            
            # Best method analysis
            best_method = method_comparison.get('best_method', '')
            if best_method == 'method_1':
                agreement_analysis['method_1_wins'] += 1
            elif best_method == 'method_2':
                agreement_analysis['method_2_wins'] += 1
            elif best_method == 'method_3':
                agreement_analysis['method_3_wins'] += 1
        
        # Print results
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.SUCCESS('METHOD ACCURACY'))
        self.stdout.write('=' * 70)
        
        for method_key, stats in method_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                avg_conf = sum(stats['confidences']) / len(stats['confidences']) if stats['confidences'] else 0
                method_name = {
                    'method_1': 'Method 1: Deep Learning Model',
                    'method_2': 'Method 2: Statistical Pattern Analysis',
                    'method_3': 'Method 3: Metadata & Heuristic Analysis',
                    'final': 'Final Combined Result'
                }.get(method_key, method_key)
                
                self.stdout.write(f"\n{method_name}:")
                self.stdout.write(f"  Accuracy: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
                self.stdout.write(f"  Avg Confidence: {avg_conf*100:.1f}%")
        
        # Agreement analysis
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.SUCCESS('METHOD AGREEMENT'))
        self.stdout.write('=' * 70)
        total_analyzed = agreement_analysis['unanimous_agreement'] + \
                        agreement_analysis['majority_agreement'] + \
                        agreement_analysis['disagreement']
        
        if total_analyzed > 0:
            self.stdout.write(f"\nUnanimous Agreement: {agreement_analysis['unanimous_agreement']} ({agreement_analysis['unanimous_agreement']/total_analyzed*100:.1f}%)")
            self.stdout.write(f"Majority Agreement: {agreement_analysis['majority_agreement']} ({agreement_analysis['majority_agreement']/total_analyzed*100:.1f}%)")
            self.stdout.write(f"Disagreement: {agreement_analysis['disagreement']} ({agreement_analysis['disagreement']/total_analyzed*100:.1f}%)")
        
        # Best method frequency
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.SUCCESS('BEST METHOD DISTRIBUTION'))
        self.stdout.write('=' * 70)
        total_wins = agreement_analysis['method_1_wins'] + \
                    agreement_analysis['method_2_wins'] + \
                    agreement_analysis['method_3_wins']
        
        if total_wins > 0:
            self.stdout.write(f"\nMethod 1 (Deep Learning) selected as best: {agreement_analysis['method_1_wins']} ({agreement_analysis['method_1_wins']/total_wins*100:.1f}%)")
            self.stdout.write(f"Method 2 (Statistical) selected as best: {agreement_analysis['method_2_wins']} ({agreement_analysis['method_2_wins']/total_wins*100:.1f}%)")
            self.stdout.write(f"Method 3 (Metadata) selected as best: {agreement_analysis['method_3_wins']} ({agreement_analysis['method_3_wins']/total_wins*100:.1f}%)")
        
        # Issues found
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.WARNING('DETECTION ERRORS FOUND'))
        self.stdout.write('=' * 70)
        
        error_by_method = {}
        for issue in issues:
            method = issue['method']
            if method not in error_by_method:
                error_by_method[method] = []
            error_by_method[method].append(issue)
        
        for method, errors in error_by_method.items():
            self.stdout.write(f"\n{method}: {len(errors)} errors")
            for error in errors[:5]:  # Show first 5
                self.stdout.write(f"  Image {error['image_id']}: Predicted {error['predicted']}, Actual {error['actual']}, Confidence {error['confidence']*100:.1f}%")
        
        # Recommendations
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.SUCCESS('RECOMMENDATIONS'))
        self.stdout.write('=' * 70)
        
        # Calculate method accuracies for recommendations
        m1_acc = (method_stats['method_1']['correct'] / method_stats['method_1']['total'] * 100) if method_stats['method_1']['total'] > 0 else 0
        m2_acc = (method_stats['method_2']['correct'] / method_stats['method_2']['total'] * 100) if method_stats['method_2']['total'] > 0 else 0
        m3_acc = (method_stats['method_3']['correct'] / method_stats['method_3']['total'] * 100) if method_stats['method_3']['total'] > 0 else 0
        
        accuracies = [
            ('Method 1 (Deep Learning)', m1_acc, method_stats['method_1']['total']),
            ('Method 2 (Statistical)', m2_acc, method_stats['method_2']['total']),
            ('Method 3 (Metadata)', m3_acc, method_stats['method_3']['total'])
        ]
        
        accuracies.sort(key=lambda x: x[1], reverse=True)
        
        self.stdout.write(f"\n1. Best Performing Method: {accuracies[0][0]} ({accuracies[0][1]:.1f}% accuracy)")
        self.stdout.write(f"2. Second Best: {accuracies[1][0]} ({accuracies[1][1]:.1f}% accuracy)")
        self.stdout.write(f"3. Third Best: {accuracies[2][0]} ({accuracies[2][1]:.1f}% accuracy)")
        
        # Suggest weights based on accuracy
        if all(acc[2] > 0 for acc in accuracies):
            total_acc = sum(acc[1] for acc in accuracies)
            if total_acc > 0:
                self.stdout.write("\nSuggested Weight Distribution:")
                for i, (name, acc, _) in enumerate(accuracies):
                    weight = acc / total_acc
                    method_key = ['method_1', 'method_2', 'method_3'][i]
                    self.stdout.write(f"  {name}: {weight:.2f} ({(weight*100):.1f}%)")
        
        self.stdout.write('\n' + '=' * 70 + '\n')

