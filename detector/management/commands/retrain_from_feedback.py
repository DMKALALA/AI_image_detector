"""
Retrain models using user feedback data
Converts feedback into training samples and fine-tunes models
"""

from django.core.management.base import BaseCommand
from detector.models import ImageUpload
from pathlib import Path
import json
import shutil
from PIL import Image


class Command(BaseCommand):
    help = 'Retrain models using user feedback data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--min-feedback',
            type=int,
            default=10,
            help='Minimum feedback samples required to retrain (default: 10)'
        )
        parser.add_argument(
            '--models',
            type=str,
            choices=['hf', 'all'],
            default='hf',
            help='Which models to retrain (hf=HuggingFace only, all=all models)'
        )

    def handle(self, *args, **options):
        min_samples = options['min_feedback']
        
        # Get images with feedback
        correct_feedback = ImageUpload.objects.filter(user_feedback='correct')
        incorrect_feedback = ImageUpload.objects.filter(user_feedback='incorrect')
        
        self.stdout.write(f'Found {correct_feedback.count()} confirmed correct')
        self.stdout.write(f'Found {incorrect_feedback.count()} confirmed incorrect')
        
        total_feedback = correct_feedback.count() + incorrect_feedback.count()
        
        if total_feedback < min_samples:
            self.stdout.write(self.style.WARNING(
                f'Not enough feedback samples! Have {total_feedback}, need {min_samples}'
            ))
            self.stdout.write('Keep collecting feedback and try again later.')
            return
        
        # Create feedback training directory
        feedback_dir = Path('feedback_training_data')
        feedback_dir.mkdir(exist_ok=True)
        
        (feedback_dir / 'ai_images').mkdir(exist_ok=True)
        (feedback_dir / 'real_images').mkdir(exist_ok=True)
        
        # Process correct feedback (trust our prediction)
        for upload in correct_feedback:
            if upload.image and upload.image.path:
                try:
                    img = Image.open(upload.image.path)
                    target_dir = 'ai_images' if upload.is_ai_generated else 'real_images'
                    filename = f'feedback_{upload.pk}_{Path(upload.image.path).name}'
                    img.save(feedback_dir / target_dir / filename)
                except Exception as e:
                    self.stdout.write(f'Could not process {upload.pk}: {e}')
        
        # Process incorrect feedback (use opposite of our prediction)
        for upload in incorrect_feedback:
            if upload.image and upload.image.path:
                try:
                    img = Image.open(upload.image.path)
                    # Flip the label since we were wrong
                    target_dir = 'real_images' if upload.is_ai_generated else 'ai_images'
                    filename = f'feedback_{upload.pk}_{Path(upload.image.path).name}'
                    img.save(feedback_dir / target_dir / filename)
                except Exception as e:
                    self.stdout.write(f'Could not process {upload.pk}: {e}')
        
        # Count images
        ai_count = len(list((feedback_dir / 'ai_images').glob('*.jpg'))) + \
                   len(list((feedback_dir / 'ai_images').glob('*.png')))
        real_count = len(list((feedback_dir / 'real_images').glob('*.jpg'))) + \
                     len(list((feedback_dir / 'real_images').glob('*.png')))
        
        self.stdout.write(self.style.SUCCESS(
            f'\n✓ Created feedback training dataset:'
        ))
        self.stdout.write(f'  AI images: {ai_count}')
        self.stdout.write(f'  Real images: {real_count}')
        self.stdout.write(f'  Total: {ai_count + real_count}')
        self.stdout.write(f'  Location: {feedback_dir}/')
        
        # Create training manifest
        dataset = []
        for img_path in (feedback_dir / 'ai_images').glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                dataset.append({
                    'path': str(img_path),
                    'label': 1,
                    'label_name': 'ai',
                    'source': 'user_feedback'
                })
        
        for img_path in (feedback_dir / 'real_images').glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                dataset.append({
                    'path': str(img_path),
                    'label': 0,
                    'label_name': 'real',
                    'source': 'user_feedback'
                })
        
        # Save manifest
        with open(feedback_dir / 'feedback_dataset.json', 'w') as f:
            json.dump(dataset, f, indent=2)
        
        self.stdout.write(self.style.SUCCESS(
            f'\n✓ Saved training manifest: {feedback_dir}/feedback_dataset.json'
        ))
        
        # Instructions for retraining
        self.stdout.write(self.style.SUCCESS('\n📚 Next Steps - Retrain Models:'))
        self.stdout.write('')
        self.stdout.write('1. Prepare combined dataset (GenImage + Feedback):')
        self.stdout.write('   python manage.py combine_datasets')
        self.stdout.write('')
        self.stdout.write('2. Retrain HuggingFace models:')
        self.stdout.write('   python manage.py finetune_hf_models --model all --epochs 3')
        self.stdout.write('')
        self.stdout.write('3. Models will learn from user corrections!')
        self.stdout.write(f'   Training data now includes {total_feedback} user-validated samples')
        self.stdout.write('')
        self.stdout.write(self.style.WARNING('💡 Tip: More feedback = better models!'))
        self.stdout.write(f'   Collect {max(50 - total_feedback, 0)} more samples for best results')

