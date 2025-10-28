"""
Django management command to trigger adaptive learning update
Can be run manually or via cron job
"""

from django.core.management.base import BaseCommand
from detector.adaptive_learning_service import adaptive_learning_service

class Command(BaseCommand):
    help = 'Trigger adaptive learning update based on user feedback'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force update even if not enough time has passed',
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=100,
            help='Number of recent feedback samples to analyze',
        )

    def handle(self, *args, **options):
        self.stdout.write("=" * 60)
        self.stdout.write("ADAPTIVE LEARNING UPDATE")
        self.stdout.write("=" * 60)
        
        if options['force']:
            # Temporarily force update
            adaptive_learning_service.config['last_update'] = None
        
        # Analyze feedback
        self.stdout.write("\nAnalyzing recent feedback...")
        method_performances = adaptive_learning_service.analyze_feedback(limit=options['limit'])
        
        if not method_performances:
            self.stdout.write(self.style.WARNING("Not enough feedback samples for learning"))
            return
        
        # Display current performance
        self.stdout.write("\nCurrent Method Performance:")
        for method, perf in method_performances.items():
            method_name = method.replace('_', ' ').title()
            status = self.style.SUCCESS("✓") if perf['accuracy'] > 0.6 else self.style.ERROR("✗")
            self.stdout.write(
                f"  {status} {method_name}: {perf['accuracy']*100:.1f}% "
                f"({perf['correct']}/{perf['total']}) "
                f"[Avg Confidence: {perf['avg_confidence']*100:.1f}%]"
            )
        
        # Calculate and apply updates
        self.stdout.write("\nCalculating optimal weights...")
        optimal_weights = adaptive_learning_service.calculate_optimal_weights(method_performances)
        
        if optimal_weights:
            self.stdout.write("\nOptimal Weights:")
            for method, weight in optimal_weights.items():
                method_name = method.replace('_', ' ').title()
                self.stdout.write(f"  {method_name}: {weight*100:.1f}%")
            
            # Update service
            self.stdout.write("\nUpdating detection service...")
            updated_weights = adaptive_learning_service.update_service_weights(optimal_weights)
            adaptive_learning_service.update_confidence_calibration(method_performances)
            
            if updated_weights:
                self.stdout.write("\n" + self.style.SUCCESS("✅ Update complete!"))
                self.stdout.write("\nNew Weights:")
                for method, weight in updated_weights.items():
                    method_name = method.replace('_', ' ').title()
                    self.stdout.write(f"  {method_name}: {weight*100:.1f}%")
                
                # Save config
                adaptive_learning_service.config['last_update'] = adaptive_learning_service.config.get('last_update') or 'now'
                adaptive_learning_service.save_config()
            else:
                self.stdout.write(self.style.ERROR("Failed to update service"))
        else:
            self.stdout.write(self.style.WARNING("Could not calculate optimal weights"))
        
        self.stdout.write("\n" + "=" * 60)

