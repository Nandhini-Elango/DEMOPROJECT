# 🔒 On-Device AI Reading Assistant
## On-Device Question Answering with Foundry Local + Windows AI Foundry

A demonstration application showcasing **100% on-device AI processing** using **Foundry Local** (Microsoft's on-device runtime and management layer) integrated with **Windows AI Foundry** principles. Ask questions about your documents

## 🌟 Features

- ✅ **Complete Privacy**: All AI processing happens locally on your device
- ✅ **Foundry Local Integration**: Uses Microsoft's on-device runtime and model management
- ✅ **Enterprise Model Catalog**: Access to optimized models (Phi, Qwen, Mistral, and more)
- ✅ **Hardware Acceleration**: Utilizes DirectML for GPU/NPU acceleration
- ✅ **No Internet Required**: Works completely offline (after initial setup)
- ✅ **Modern UI**: Clean, user-friendly interface built with tkinter
- ✅ **Flexible Model Support**: Works with Foundry Local or traditional ONNX models
- ✅ **Smart Fallback**: Works even without AI model (keyword-based search)

## 🏗️ Architecture

This application uses three key components:

1. **Foundry Local** - On-device runtime and management layer for hosting models
2. **ONNX Runtime** - High-performance inference engine (with DirectML)
3. **Developer Tooling** - CLI and SDK for model lifecycle management

## 🚀 Quick Start

### Prerequisites

- **Windows 10/11** (64-bit) or **Windows 11** with NPU support
- **Python 3.8 or later** ([Download Python](https://www.python.org/downloads/))
- **Foundry Local** installed (recommended) - [Installation Guide](https://aka.ms/ai-foundry)
- Compatible GPU/NPU (optional, but recommended for better performance)

### 📥 Download Required Model (Important!)

**Note**: The `model.onnx` file is not included in this repository due to its large size (~250MB).

**Quick Setup - Choose One Option:**

**Option 1: Auto-download via Script (Recommended)**
```powershell
python download_model.py
```
Select option 2: "BERT SQuAD" model when prompted.

**Option 2: Manual Download**
1. Download: [BERT SQuAD ONNX Model](https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx)
2. Rename the downloaded file to `model.onnx`
3. Place it in the project root folder

**Option 3: Use Foundry Local (No manual download needed)**
```powershell
python foundry_model_manager.py
```
This will use Foundry Local's managed models instead.

### Installation Steps

#### Option A: Using Foundry Local (Recommended)

1. **Install Foundry Local** (if not already installed)
   ```powershell
   # Follow Microsoft's installation guide
   # https://aka.ms/ai-foundry
   ```

2. **Verify Foundry Local installation**
   ```powershell
   foundry --version
   ```

1. **Navigate to project folder**
   ```powershell
   cd Project
   ```

2. **Install Python dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Download an AI model using Foundry Local**
   ```powershell
   python foundry_model_manager.py
   ```
   
   Recommended models:
   - **Phi-3.5 Mini** - Best balance of performance and size (~2.5 GB)
   - **Qwen 2.5 0.5B** - Ultra-lightweight for quick responses (~0.8 GB)
   - **Phi-4 Mini** - Latest model with improved accuracy (~4.8 GB)

6. **Run the application**
   ```powershell
   python DemoApp.py
   ```

#### Option B: Using Traditional ONNX Model (Without Foundry Local)

1. **Navigate to project folder**
   ```powershell
   cd Project
   ```

2. **Install required packages**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Download BERT model**
   ```powershell
   python download_model.py
   ```
   Select option 2: "BERT SQuAD" model (designed for Q&A)

4. **Run the application**
   ```powershell
   python DemoApp.py
   ```

## 📖 How to Use

1. **Launch the application** by running `DemoApp.py`
2. **Paste your document/text** in the context box (sample text is pre-loaded)
3. **Type your question** in the question box
4. **Click "Get Answer"** to let the AI extract the answer from your text
5. **View the answer** in the result box below

### Example Usage:

**Context (your document):**
```
Windows AI Foundry provides tools for building On-Device AI applications. 
It uses DirectML for hardware acceleration and ONNX Runtime for inference.
```

**Question:**
```
What does Windows AI Foundry use for hardware acceleration?
```

**AI Answer:**
```
DirectML
```

## 🔧 Technical Details

### Architecture Stack

```
┌─────────────────────────────────────┐
│  Application Layer (DemoApp.py)     │
│  • UI (Tkinter)                     │
│  • Q&A Logic                        │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Foundry Local Integration Layer    │
│  • Model Management                 │
│  • Runtime Orchestration            │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  ONNX Runtime + DirectML            │
│  • Inference Engine                 │
│  • Hardware Acceleration            │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Silicon (CPU / GPU / NPU)          │
│  • Device Hardware                  │
└─────────────────────────────────────┘
```

### Hardware Acceleration

The application uses **ONNX Runtime with DirectML** to leverage:
- **GPUs** (NVIDIA, AMD, Intel)
- **NPUs** (Neural Processing Units - Qualcomm, Intel)
- **CPU fallback** if no accelerator is available

### Foundry Local Benefits

1. **Centralized Model Management**: Easy model download, update, and versioning
2. **Optimized Models**: Pre-optimized for Windows hardware (CPU/GPU/NPU)
3. **Professional Tooling**: CLI and SDK for programmatic access
4. **Multi-Model Support**: Run different models for different tasks
5. **Automatic Updates**: Optional model updates through Microsoft catalog

### Privacy Architecture

```
┌─────────────────────────────────────┐
│  Your Computer (100% Local)         │
│  ┌──────────────────────────────┐  │
│  │  User Input                   │  │
│  └────────────┬─────────────────┘  │
│               ▼                     │
│  ┌──────────────────────────────┐  │
│  │  ONNX Model (On-Device)      │  │
│  │  + DirectML Acceleration      │  │
│  └────────────┬─────────────────┘  │
│               ▼                     │
│  ┌──────────────────────────────┐  │
│  │  AI Results                   │  │
│  └──────────────────────────────┘  │
│                                     │
│  🔐 No data sent to cloud/internet │
└─────────────────────────────────────┘
```

### Application Modes

1. **Foundry Local Mode** (Recommended): Uses Foundry-managed models with enterprise features
2. **ONNX Mode**: Uses traditional BERT SQuAD model with proper tokenization
3. **Fallback Mode**: Uses keyword matching when no model is available

## 📦 Project Structure

```
Project/
├── DemoApp.py                    # Main Q&A application (Foundry Local integrated)
├── foundry_model_manager.py      # Foundry Local model management CLI tool
├── foundry_runtime.py            # Foundry Local integration layer
├── download_model.py             # Legacy ONNX model download utility
├── requirements.txt              # Python dependencies
├── model.onnx                    # Traditional AI model (optional - download separately)
├── .foundry_model                # Stores Foundry Local model preference
└── README.md                     # This file
```

## 🛠️ Troubleshooting

### Foundry Local Issues

**"Foundry Local CLI not found"**
**Solution**: Install Foundry Local:
```powershell
# Visit https://aka.ms/ai-foundry for installation instructions
foundry --version  # Verify installation
```

**"Model not found in Foundry Local"**
**Solution**: Download the model first:
```powershell
python foundry_model_manager.py
# Or directly:
foundry model download phi-3.5-mini
```

**"Foundry model run failed"**
**Solution**: Check model availability and device compatibility:
```powershell
foundry model list
foundry model info phi-3.5-mini
```

### Traditional ONNX Model Issues

**"Import transformers could not be resolved"**
**Solution**: Install the transformers package:
```powershell
pip install transformers
```

**"Model file not found"**
**Solution**: Download the BERT SQuAD model:
```powershell
python download_model.py
```
Select option 2 (BERT SQuAD) for question answering.

### Performance Issues

**Performance is slow**
**Solution**: 
1. Check hardware acceleration status in the app status bar
2. For Foundry Local: Use NPU-optimized models if you have NPU hardware
3. Try a smaller model (e.g., Qwen 2.5 0.5B instead of Phi-4)

### Answers Issues

**Answers are not accurate**
**Solution**: 
1. **Foundry Local**: Try a larger model (Phi-4 Mini or Qwen 2.5 7B)
2. **ONNX Mode**: Ensure transformers library is installed
3. Try rephrasing your question to be more specific
4. Ensure the answer is actually present in the context text

## 🎯 Use Cases

- **Document Analysis**: Extract specific information from documents
- **Research Assistant**: Query research papers and articles
- **Study Helper**: Ask questions about textbook content
- **Email/Report Analysis**: Find key information quickly
- **Privacy-Critical Documents**: Process sensitive documents locally without cloud services

## 🔐 Privacy Benefits

1. **No Cloud Dependencies**: Everything runs on your device
2. **No Data Collection**: Your text never leaves your computer
3. **Offline Capable**: Works without internet connection
4. **Full Control**: You own your data and the processing

## 🚀 Next Steps & Enhancements

Want to extend this application? Try:
- **Multiple Models**: Load different Foundry Local models for different tasks
- **Model Comparison**: Compare answers from different models side-by-side
- **Batch Processing**: Process multiple questions at once
- **Web Interface**: Create a local web server with Flask/FastAPI
- **Export Functionality**: Add PDF, Word, or CSV export
- **Voice Input**: Integrate speech-to-text for voice questions
- **Document Upload**: Add support for PDF/DOCX file uploads

## 🔗 Useful Commands

### Foundry Local CLI

```powershell
# List all models
foundry model list

# Get model details
foundry model info phi-3.5-mini

# Download a model
foundry model download qwen2.5-1.5b

# Run model interactively
foundry model run phi-4-mini

# Check Foundry version
foundry --version
```

## 📚 Learn More

- [Foundry Local Documentation](https://aka.ms/ai-foundry)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [DirectML on Windows](https://docs.microsoft.com/en-us/windows/ai/directml/dml)
- [Windows AI Development](https://docs.microsoft.com/en-us/windows/ai/)
- [Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-foundry)

## 📄 License

This is a demonstration project for educational purposes.

---

**Built with ❤️ using Foundry Local + Windows AI Foundry**

*Your privacy matters. Keep AI processing on your device.*

### 🏗️ Tech Stack

- **Foundry Local** - On-device runtime and model management
- **ONNX Runtime** - High-performance inference engine
- **DirectML** - Hardware acceleration layer
- **Python** - Application framework
- **Tkinter** - User interface
