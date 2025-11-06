"""
ONNX Model Download Utility for On-Device AI Application
Downloads AI models from ONNX Model Zoo for on-device inference
"""

import urllib.request
import os
import sys

def download_with_progress(url, filename):
    """Download file with progress indication"""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        bars = int(percent / 2)
        sys.stdout.write(f'\r[{"=" * bars}{" " * (50 - bars)}] {percent:.1f}%')
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filename, progress_hook)
    print()  # New line after progress bar

def main():
    """Main model download function"""
    print("=" * 70)
    print("  ONNX Model Downloader - On-Device AI Application")
    print("=" * 70)
    print("\nThis will download an AI model for on-device inference.")
    print("The model will be used locally - no cloud processing required!\n")
    
    # Model options
    models = [
        {
            "name": "RoBERTa Sequence Classification (Recommended)",
            "url": "https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/roberta/model/roberta-sequence-classification-9.onnx",
            "size": "~450 MB",
            "description": "Good for text classification and understanding"
        },
        {
            "name": "BERT SQuAD (Large)",
            "url": "https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx",
            "size": "~400 MB",
            "description": "Optimized for question answering tasks"
        }
    ]
    
    # Display options
    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   Size: {model['size']}")
        print(f"   Description: {model['description']}")
    
    print(f"\n{len(models) + 1}. Skip download (use fallback mode)")
    
    # User selection
    print("\n" + "-" * 70)
    try:
        choice = input(f"Select model to download (1-{len(models) + 1}): ").strip()
        
        if not choice or choice == str(len(models) + 1):
            print("\n✓ Skipping model download.")
            print("  The application will run in fallback mode.")
            print("  You can download a model later by running this script again.")
            return
        
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(models):
            print("Invalid selection. Using default model.")
            choice_idx = 0
            
    except (ValueError, KeyboardInterrupt):
        print("\nUsing default model...")
        choice_idx = 0
    
    selected_model = models[choice_idx]
    
    print("\n" + "=" * 70)
    print(f"Downloading: {selected_model['name']}")
    print(f"Size: {selected_model['size']}")
    print("=" * 70)
    
    # Check if model already exists
    if os.path.exists("model.onnx"):
        overwrite = input("\nModel file already exists. Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Download cancelled.")
            return
    
    # Download the model
    try:
        print("\nDownloading... This may take a few minutes depending on your connection.\n")
        download_with_progress(selected_model['url'], "model.onnx")
        
        # Verify download
        file_size = os.path.getsize("model.onnx") / (1024 * 1024)  # Convert to MB
        
        print("\n" + "=" * 70)
        print("✓ Model downloaded successfully!")
        print(f"✓ File: model.onnx")
        print(f"✓ Size: {file_size:.1f} MB")
        print("=" * 70)
        print("\nYou can now run the application:")
        print("  python DemoApp.py")
        print("\nThe model will be used for on-device AI inference.")
        print("Your data stays private and never leaves your computer! 🔒")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify you have enough disk space")
        print("3. Try running the script again")
        print("\nNote: The application can still run in fallback mode without a model.")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code if exit_code else 0)
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)