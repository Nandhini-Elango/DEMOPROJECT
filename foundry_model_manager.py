"""
Foundry Local Model Manager
Uses Foundry Local CLI and runtime for model lifecycle management
"""

import subprocess
import json
import sys
import os

class FoundryModelManager:
    """Manage models using Foundry Local CLI"""
    
    def __init__(self):
        self.verify_foundry_installed()
    
    def verify_foundry_installed(self):
        """Check if Foundry Local is installed"""
        try:
            result = subprocess.run(
                ['foundry', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Foundry Local installed: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ Foundry Local CLI not found!")
            print("Please install Foundry Local from:")
            print("https://aka.ms/ai-foundry")
            sys.exit(1)
    
    def list_models(self):
        """List all available models in Foundry Local catalog"""
        try:
            result = subprocess.run(
                ['foundry', 'model', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error listing models: {e}")
            return None
    
    def get_available_models(self):
        """Parse and return available models for Q&A tasks"""
        output = self.list_models()
        if not output:
            return []
        
        # Models suitable for question answering or text tasks
        qa_models = [
            {
                'alias': 'phi-3.5-mini',
                'name': 'Phi-3.5 Mini',
                'description': 'Small, efficient model for text understanding',
                'device': 'NPU/CPU',
                'size': '~2.5 GB'
            },
            {
                'alias': 'phi-3-mini-128k',
                'name': 'Phi-3 Mini 128k',
                'description': 'Phi-3 with large context window',
                'device': 'NPU/CPU',
                'size': '~2.5 GB'
            },
            {
                'alias': 'qwen2.5-0.5b',
                'name': 'Qwen 2.5 0.5B',
                'description': 'Ultra-lightweight model for basic tasks',
                'device': 'CPU',
                'size': '~0.8 GB'
            },
            {
                'alias': 'qwen2.5-1.5b',
                'name': 'Qwen 2.5 1.5B',
                'description': 'Balanced performance and size',
                'device': 'NPU/CPU',
                'size': '~1.8 GB'
            },
            {
                'alias': 'phi-4-mini',
                'name': 'Phi-4 Mini',
                'description': 'Latest Phi model with improved performance',
                'device': 'CPU',
                'size': '~4.8 GB'
            }
        ]
        
        return qa_models
    
    def download_model(self, model_alias):
        """Download a model using Foundry Local CLI"""
        print(f"\n📥 Downloading model: {model_alias}")
        print("This may take a few minutes...")
        
        try:
            # Use foundry CLI to download the model
            result = subprocess.run(
                ['foundry', 'model', 'download', model_alias],
                check=True,
                text=True
            )
            print(f"✓ Model {model_alias} downloaded successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error downloading model: {e}")
            return False
    
    def get_model_info(self, model_alias):
        """Get detailed information about a model"""
        try:
            result = subprocess.run(
                ['foundry', 'model', 'info', model_alias],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return None


def main():
    """Interactive model management using Foundry Local"""
    print("=" * 70)
    print("  Foundry Local Model Manager")
    print("  On-Device AI Model Management")
    print("=" * 70)
    print()
    
    manager = FoundryModelManager()
    
    # Get available models
    models = manager.get_available_models()
    
    print("Available models optimized for this application:")
    print()
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']} ({model['alias']})")
        print(f"   Description: {model['description']}")
        print(f"   Device: {model['device']}")
        print(f"   Size: {model['size']}")
        print()
    
    print(f"{len(models) + 1}. View all models in catalog")
    print(f"{len(models) + 2}. Exit")
    
    print("-" * 70)
    
    try:
        choice = input(f"Select an option (1-{len(models) + 2}): ").strip()
        
        if not choice or choice == str(len(models) + 2):
            print("Exiting...")
            return
        
        if choice == str(len(models) + 1):
            print("\nFull model catalog:")
            print("=" * 70)
            print(manager.list_models())
            return
        
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(models):
            print("Invalid selection.")
            return
        
        selected_model = models[choice_idx]
        
        print("\n" + "=" * 70)
        print(f"Selected: {selected_model['name']}")
        print("=" * 70)
        
        # Ask to download
        download = input(f"\nDownload {selected_model['alias']}? (y/N): ").strip().lower()
        
        if download == 'y':
            if manager.download_model(selected_model['alias']):
                # Save the selected model alias for the app to use
                with open('.foundry_model', 'w') as f:
                    f.write(selected_model['alias'])
                
                print("\n" + "=" * 70)
                print("✓ Model ready!")
                print(f"✓ Model alias: {selected_model['alias']}")
                print("=" * 70)
                print("\nYou can now run the application:")
                print("  python DemoApp.py")
                print("\nThe model will be managed by Foundry Local.")
                print("All processing happens on-device! 🔒")
        else:
            print("Download cancelled.")
    
    except (ValueError, KeyboardInterrupt):
        print("\n\nOperation cancelled.")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code if exit_code else 0)
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(1)
