"""
Foundry Local Runtime Integration
Provides interface to Foundry Local's on-device runtime and inference
"""

import subprocess
import json
import os
import tempfile
import onnxruntime as ort


class FoundryLocalRuntime:
    """
    Integration layer for Foundry Local runtime
    Manages model loading and inference using Foundry Local infrastructure
    """
    
    def __init__(self, model_alias=None):
        """
        Initialize Foundry Local runtime
        
        Args:
            model_alias: The Foundry model alias (e.g., 'phi-3.5-mini')
        """
        self.model_alias = model_alias
        self.session = None
        self.using_foundry = False
        self.model_path = None
    
    def get_model_path_from_foundry(self, model_alias):
        """
        Get the local path to a Foundry-managed model
        
        Args:
            model_alias: Model alias in Foundry catalog
            
        Returns:
            Path to the ONNX model file, or None if not available
        """
        try:
            # Try to get model info from Foundry
            result = subprocess.run(
                ['foundry', 'model', 'info', model_alias, '--json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            info = json.loads(result.stdout)
            # Extract model path from info
            # This is a simplified version - actual implementation depends on Foundry's API
            if 'path' in info:
                return info['path']
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            pass
        
        return None
    
    def run_foundry_model_inference(self, model_alias, prompt):
        """
        Run inference using Foundry Local's runtime
        
        Args:
            model_alias: Model to use
            prompt: Input prompt/question
            
        Returns:
            Model output text
        """
        try:
            # Use Foundry CLI to run inference
            result = subprocess.run(
                ['foundry', 'model', 'run', model_alias, '--prompt', prompt],
                capture_output=True,
                text=True,
                check=True,
                timeout=120
            )
            
            return result.stdout.strip()
            
        except subprocess.CalledProcessError as e:
            print(f"Foundry inference error: {e}")
            return None
        except subprocess.TimeoutExpired:
            print("Foundry inference timeout")
            return None
    
    def load_foundry_model(self, model_alias):
        """
        Load a model from Foundry Local catalog
        
        Args:
            model_alias: Model alias (e.g., 'phi-3.5-mini')
            
        Returns:
            True if loaded successfully, False otherwise
        """
        print(f"Loading model from Foundry Local: {model_alias}")
        
        # Check if model is available
        try:
            result = subprocess.run(
                ['foundry', 'model', 'info', model_alias],
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"✓ Model {model_alias} found in Foundry Local")
            self.model_alias = model_alias
            self.using_foundry = True
            return True
            
        except subprocess.CalledProcessError:
            print(f"✗ Model {model_alias} not found in Foundry Local")
            print("  Run: python foundry_model_manager.py to download")
            return False
    
    def load_onnx_model_direct(self, model_path):
        """
        Load ONNX model directly using ONNX Runtime
        Fallback for traditional model loading
        
        Args:
            model_path: Path to .onnx file
            
        Returns:
            True if loaded successfully
        """
        try:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.model_path = model_path
            self.using_foundry = False
            
            active_provider = self.session.get_providers()[0]
            print(f"✓ Loaded ONNX model directly: {model_path}")
            print(f"✓ Using: {active_provider}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading ONNX model: {e}")
            return False
    
    def inference_qa(self, question, context):
        """
        Run question answering inference
        
        Args:
            question: Question to answer
            context: Context text to search for answer
            
        Returns:
            Answer text or None
        """
        if self.using_foundry:
            # Use Foundry Local runtime for inference
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            return self.run_foundry_model_inference(self.model_alias, prompt)
        
        elif self.session:
            # Use ONNX Runtime directly
            # This requires model-specific input preparation
            # Implementation depends on the specific model format
            return None  # Placeholder - implement based on model type
        
        return None
    
    def get_status(self):
        """Get runtime status information"""
        status = {
            'using_foundry': self.using_foundry,
            'model_alias': self.model_alias,
            'model_path': self.model_path,
            'session_active': self.session is not None
        }
        return status


def check_foundry_available():
    """Check if Foundry Local is installed and available"""
    try:
        result = subprocess.run(
            ['foundry', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, None


def get_saved_model_preference():
    """Get the user's saved model preference"""
    if os.path.exists('.foundry_model'):
        with open('.foundry_model', 'r') as f:
            return f.read().strip()
    return None
