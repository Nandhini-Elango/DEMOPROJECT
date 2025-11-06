"""
On-Device AI Application with Windows AI Foundry + Foundry Local
On-Device Question Answering Assistant - All processing happens locally on your device
Now enhanced with Foundry Local runtime and management layer
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import onnxruntime as ort
import numpy as np
import os
from datetime import datetime
import warnings

# Import Foundry Local integration
from foundry_runtime import FoundryLocalRuntime, check_foundry_available, get_saved_model_preference

# Suppress transformers warning about PyTorch/TensorFlow (we only need tokenizer)
warnings.filterwarnings('ignore', message='.*PyTorch.*TensorFlow.*Flax.*')

class PrivacyFirstAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("On-Device AI Assistant - Foundry Local")
        self.root.geometry("900x700")
        self.root.configure(bg='#f5f7fa')
        
        # Set minimum window size
        self.root.minsize(800, 600)
        
        # Check Foundry Local availability
        self.foundry_available, self.foundry_version = check_foundry_available()
        
        # Initialize model
        self.session = None
        self.model_loaded = False
        self.tokenizer = None
        self.foundry_runtime = None
        self.using_foundry = False
        
        self.load_model()
        
        # Create UI
        self.create_ui()
        
    def load_model(self):
        """Load model using Foundry Local or fallback to direct ONNX loading"""
        print("=" * 70)
        print("INITIALIZING AI MODEL")
        print("=" * 70)
        
        # Priority 1: Try Foundry Local
        if self.foundry_available:
            print(f"✓ Foundry Local detected: {self.foundry_version}")
            
            # Check for saved model preference
            preferred_model = get_saved_model_preference()
            
            if preferred_model:
                print(f"Loading preferred model: {preferred_model}")
                self.foundry_runtime = FoundryLocalRuntime(preferred_model)
                
                if self.foundry_runtime.load_foundry_model(preferred_model):
                    self.model_loaded = True
                    self.using_foundry = True
                    print("✓ Using Foundry Local runtime for inference")
                    print("=" * 70)
                    return
                else:
                    print("⚠ Foundry model not available, trying alternatives...")
        else:
            print("⚠ Foundry Local not detected")
            print("  For enhanced features, install Foundry Local:")
            print("  https://aka.ms/ai-foundry")
        
        # Priority 2: Try traditional model.onnx
        if os.path.exists("model.onnx"):
            print("\nLoading traditional ONNX model (model.onnx)...")
            self.foundry_runtime = FoundryLocalRuntime()
            
            if self.foundry_runtime.load_onnx_model_direct("model.onnx"):
                # Try to load BERT tokenizer for the ONNX model
                try:
                    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
                    self.session = ort.InferenceSession("model.onnx", providers=providers)
                    self.model_loaded = True
                    self.using_foundry = False  # Using ONNX, not Foundry
                    
                    active_provider = self.session.get_providers()[0]
                    
                    # Try to load tokenizer
                    try:
                        import sys
                        import io
                        
                        old_stderr = sys.stderr
                        sys.stderr = io.StringIO()
                        
                        from transformers import BertTokenizer
                        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
                        
                        sys.stderr = old_stderr
                        
                        print("✓ Model and tokenizer loaded successfully!")
                        print(f"✓ Using: {active_provider}")
                    except ImportError:
                        sys.stderr = old_stderr
                        print("✓ Model loaded successfully!")
                        print(f"✓ Using: {active_provider}")
                        print("⚠ Tokenizer not available (install transformers)")
                        self.tokenizer = None
                    
                    print("=" * 70)
                    return
                    
                except Exception as e:
                    print(f"⚠ Error loading model.onnx: {e}")
        
        # No model available
        print("\n⚠ No AI model loaded")
        print("  Options:")
        print("  1. Use Foundry Local: python foundry_model_manager.py")
        print("  2. Download ONNX model: python download_model.py")
        print("  Application will run in fallback mode.")
        print("=" * 70)
    
    def create_ui(self):
        """Create the modern, beautiful user interface"""
        # Configure root window
        self.root.configure(bg='#f5f7fa')
        
        # Modern gradient-style header
        header_frame = tk.Frame(self.root, bg='#667eea', pady=15)
        header_frame.pack(fill='x')
        
        # Main title with icon
        title_label = tk.Label(
            header_frame,
            text="🔒 On-Device AI Assistant",
            font=('Segoe UI', 18, 'bold'),
            bg='#667eea',
            fg='white'
        )
        title_label.pack()
        
        # Subtitle with better spacing
        subtitle_label = tk.Label(
            header_frame,
            text="Ask Questions  •  On-Device Processing",
            font=('Segoe UI', 9),
            bg='#667eea',
            fg='#e0e7ff'
        )
        subtitle_label.pack(pady=(3, 0))
        
        # Modern status bar with rounded corners and gradient
        status_frame = tk.Frame(self.root, bg='#f5f7fa', pady=8)
        status_frame.pack(fill='x')
        
        # Determine status text and colors
        if self.using_foundry:
            status_text = f"✓ Foundry Local Active  •  Model: {self.foundry_runtime.model_alias}"
            status_bg = '#10b981'
            status_fg = 'white'
        elif self.model_loaded and self.tokenizer:
            provider = self.session.get_providers()[0]
            status_text = f"✓ AI Model Ready  •  {provider}"
            status_bg = '#10b981'
            status_fg = 'white'
        elif self.model_loaded:
            status_text = "⚠ Model Loaded  •  Enhanced features available with transformers"
            status_bg = '#f59e0b'
            status_fg = 'white'
        else:
            status_text = "⚠ Fallback Mode  •  Basic keyword matching active"
            status_bg = '#ef4444'
            status_fg = 'white'
        
        # Create a canvas for rounded status badge with proper width
        canvas_width = 800
        status_canvas = tk.Canvas(status_frame, width=canvas_width, height=40, bg='#f5f7fa', highlightthickness=0)
        status_canvas.pack()
        
        # Draw rounded rectangle for status - centered
        badge_width = 600
        badge_height = 30
        radius = 15
        x_center = canvas_width // 2
        y_center = 20
        x1, y1 = x_center - badge_width//2, y_center - badge_height//2
        x2, y2 = x_center + badge_width//2, y_center + badge_height//2
        
        # Draw rounded rectangle
        status_canvas.create_arc(x1, y1, x1+2*radius, y1+2*radius, start=90, extent=90, fill=status_bg, outline='')
        status_canvas.create_arc(x2-2*radius, y1, x2, y1+2*radius, start=0, extent=90, fill=status_bg, outline='')
        status_canvas.create_arc(x1, y2-2*radius, x1+2*radius, y2, start=180, extent=90, fill=status_bg, outline='')
        status_canvas.create_arc(x2-2*radius, y2-2*radius, x2, y2, start=270, extent=90, fill=status_bg, outline='')
        status_canvas.create_rectangle(x1+radius, y1, x2-radius, y2, fill=status_bg, outline='')
        status_canvas.create_rectangle(x1, y1+radius, x2, y2-radius, fill=status_bg, outline='')
        
        # Add text on top
        status_canvas.create_text(
            x_center, y_center,
            text=status_text,
            font=('Segoe UI', 9, 'bold'),
            fill=status_fg
        )
        
        # Create scrollable container for main content
        container = tk.Frame(self.root, bg='#f5f7fa')
        container.pack(fill='both', expand=True)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(container, bg='#f5f7fa', highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient='vertical', command=canvas.yview)
        
        # Create main frame inside canvas
        main_frame = tk.Frame(canvas, bg='#f5f7fa', padx=20, pady=10)
        
        # Configure scrolling and canvas width update
        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
        
        def _on_canvas_configure(event):
            # Update the window width to match canvas width
            canvas.itemconfig(canvas_window, width=event.width)
        
        main_frame.bind('<Configure>', _on_frame_configure)
        canvas.bind('<Configure>', _on_canvas_configure)
        
        # Create window in canvas and store reference
        canvas_window = canvas.create_window((0, 0), window=main_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Context section with modern styling
        context_label = tk.Label(
            main_frame,
            text="📄  Your Document / Context",
            font=('Segoe UI', 11, 'bold'),
            bg='#f5f7fa',
            fg='#1f2937'
        )
        context_label.pack(anchor='w', pady=(0, 5))
        
        # Context box with modern styling
        context_container = tk.Frame(main_frame, bg='#ffffff', relief='flat', bd=0, padx=2, pady=2)
        context_container.pack(fill='both', expand=True, pady=(0, 8))
        
        self.context_box = scrolledtext.ScrolledText(
            context_container,
            height=5,
            width=70,
            font=('Segoe UI', 10),
            wrap=tk.WORD,
            relief='flat',
            borderwidth=0,
            bg='#ffffff',
            fg='#374151',
            insertbackground='#667eea',
            selectbackground='#c7d2fe',
            selectforeground='#1e293b',
            padx=15,
            pady=15
        )
        self.context_box.pack(fill='both', expand=True)
        
        # Sample context
        sample_context = """Windows AI Foundry is Microsoft's framework for building On-Device AI applications that run entirely on Windows devices. It leverages DirectML for hardware acceleration, allowing developers to utilize GPUs, NPUs, and other AI accelerators. The framework uses ONNX Runtime as its inference engine, which supports models trained in various frameworks like PyTorch and TensorFlow.

Key benefits include complete privacy since all data processing happens locally on the user's device, no internet connection required after initial setup, and cost-effectiveness as there are no cloud API fees. DirectML works with hardware from NVIDIA, AMD, Intel, and Qualcomm, making it versatile across different Windows devices.

Windows AI Foundry is particularly useful for scenarios requiring data privacy, such as processing sensitive documents, medical records, or personal information. Applications can run offline, making them reliable in environments with limited connectivity."""
        
        self.context_box.insert('1.0', sample_context)
        
        # Question section with modern card
        question_label = tk.Label(
            main_frame,
            text="❓  Your Question",
            font=('Segoe UI', 11, 'bold'),
            bg='#f5f7fa',
            fg='#1f2937'
        )
        question_label.pack(anchor='w', pady=(0, 5))
        
        question_container = tk.Frame(main_frame, bg='#ffffff', relief='flat', bd=0, padx=2, pady=2)
        question_container.pack(fill='x', pady=(0, 8))
        
        self.question_box = scrolledtext.ScrolledText(
            question_container,
            height=2,
            width=70,
            font=('Segoe UI', 10),
            wrap=tk.WORD,
            relief='flat',
            borderwidth=0,
            bg='#ffffff',
            fg='#374151',
            insertbackground='#667eea',
            selectbackground='#c7d2fe',
            selectforeground='#1e293b',
            padx=15,
            pady=12
        )
        self.question_box.pack(fill='x')
        
        # Sample question
        sample_question = "What does Windows AI Foundry use for hardware acceleration?"
        self.question_box.insert('1.0', sample_question)
        
        # Button frame with modern buttons
        button_frame = tk.Frame(main_frame, bg='#f5f7fa')
        button_frame.pack(pady=8)
        
        # Create modern gradient button using Canvas
        self.answer_btn = tk.Button(
            button_frame,
            text="🤖  Get Answer",
            command=self.answer_question,
            font=('Segoe UI', 10, 'bold'),
            bg='#667eea',
            fg='white',
            activebackground='#5568d3',
            activeforeground='white',
            padx=30,
            pady=10,
            relief='flat',
            cursor='hand2',
            borderwidth=0
        )
        self.answer_btn.pack(side='left', padx=8)
        
        # Add hover effects
        self.answer_btn.bind('<Enter>', lambda e: self.answer_btn.config(bg='#5568d3'))
        self.answer_btn.bind('<Leave>', lambda e: self.answer_btn.config(bg='#667eea'))
        
        # Modern secondary button
        clear_btn = tk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_text,
            font=('Segoe UI', 10),
            bg='#e5e7eb',
            fg='#374151',
            activebackground='#d1d5db',
            activeforeground='#1f2937',
            padx=25,
            pady=10,
            relief='flat',
            cursor='hand2',
            borderwidth=0
        )
        clear_btn.pack(side='left', padx=8)
        
        # Add hover effects for clear button
        clear_btn.bind('<Enter>', lambda e: clear_btn.config(bg='#d1d5db'))
        clear_btn.bind('<Leave>', lambda e: clear_btn.config(bg='#e5e7eb'))
        
        # Output section with modern card styling
        output_label = tk.Label(
            main_frame,
            text="💡  Answer",
            font=('Segoe UI', 11, 'bold'),
            bg='#f5f7fa',
            fg='#1f2937'
        )
        output_label.pack(anchor='w', pady=(8, 5))
        
        result_container = tk.Frame(main_frame, bg='#f0fdf4', relief='flat', bd=0, padx=2, pady=2)
        result_container.pack(fill='both', expand=True)
        
        self.result_box = scrolledtext.ScrolledText(
            result_container,
            height=5,
            width=70,
            font=('Segoe UI', 10),
            wrap=tk.WORD,
            relief='flat',
            borderwidth=0,
            bg='#f0fdf4',
            fg='#065f46',
            insertbackground='#10b981',
            selectbackground='#d1fae5',
            selectforeground='#064e3b',
            padx=15,
            pady=15
        )
        self.result_box.pack(fill='both', expand=True)
        
        # Modern footer with gradient
        footer_frame = tk.Frame(self.root, bg='#1f2937', pady=8)
        footer_frame.pack(fill='x', side='bottom')
        
        footer_label = tk.Label(
            footer_frame,
            text="🔐  On-Device  •  Windows AI Foundry",
            font=('Segoe UI', 8),
            bg='#1f2937',
            fg='#9ca3af'
        )
        footer_label.pack()
    
    def answer_question(self):
        """Process context and question to generate answer"""
        context = self.context_box.get("1.0", tk.END).strip()
        question = self.question_box.get("1.0", tk.END).strip()
        
        if not context:
            messagebox.showwarning("No Context", "Please paste some text/document first.")
            return
        
        if not question:
            messagebox.showwarning("No Question", "Please enter a question.")
            return
        
        # Disable button during processing
        self.answer_btn.config(state='disabled', text="Processing...")
        self.root.update()
        
        try:
            if self.using_foundry and self.foundry_runtime:
                # Use Foundry Local runtime
                answer = self.foundry_answer(context, question)
            elif self.model_loaded and self.tokenizer:
                # Use traditional BERT model
                answer = self.ai_answer(context, question)
            else:
                # Fallback mode
                answer = self.fallback_answer(context, question)
            
            # Display result
            self.result_box.delete('1.0', tk.END)
            self.result_box.insert('1.0', answer)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Error during Q&A: {e}")
        
        finally:
            # Re-enable button
            self.answer_btn.config(state='normal', text="🤖 Get Answer")
    
    def foundry_answer(self, context, question):
        """Use Foundry Local runtime for question answering"""
        try:
            # Foundry Local models are chat-completion models
            # Format the prompt for Q&A
            prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""
            
            print(f"\nQuerying Foundry Local model: {self.foundry_runtime.model_alias}")
            
            # Run inference through Foundry Local
            answer = self.foundry_runtime.run_foundry_model_inference(
                self.foundry_runtime.model_alias, 
                prompt
            )
            
            if answer:
                # Clean up the answer
                answer = answer.strip()
                return f"{answer}\n\n✓ Answer generated using Foundry Local runtime\n✓ Model: {self.foundry_runtime.model_alias}\n✓ All processing on-device"
            else:
                return "Unable to generate answer using Foundry Local. Please try again or use fallback mode."
                
        except Exception as e:
            print(f"Foundry Local inference error: {e}")
            import traceback
            traceback.print_exc()
            return self.fallback_answer(context, question)
    
    def ai_answer(self, context, question):
        """Use BERT model for question answering"""
        try:
            # Tokenize question and context together
            inputs = self.tokenizer.encode_plus(
                question,
                context,
                add_special_tokens=True,
                return_tensors="np",
                max_length=384,
                truncation=True,
                padding='max_length'
            )
            
            # Prepare inputs for ONNX model
            input_ids = inputs['input_ids'].astype(np.int64)
            attention_mask = inputs['attention_mask'].astype(np.int64)
            token_type_ids = inputs['token_type_ids'].astype(np.int64)
            
            # Get model input names
            input_names = [inp.name for inp in self.session.get_inputs()]
            
            # Build input dictionary based on model's expected inputs
            onnx_inputs = {}
            if 'input_ids' in input_names or 'input_ids:0' in input_names:
                key = 'input_ids:0' if 'input_ids:0' in input_names else 'input_ids'
                onnx_inputs[key] = input_ids
            if 'attention_mask' in input_names or 'input_mask:0' in input_names:
                key = 'input_mask:0' if 'input_mask:0' in input_names else 'attention_mask'
                onnx_inputs[key] = attention_mask
            if 'token_type_ids' in input_names or 'segment_ids:0' in input_names:
                key = 'segment_ids:0' if 'segment_ids:0' in input_names else 'token_type_ids'
                onnx_inputs[key] = token_type_ids
            
            # Handle unique_ids if required
            if 'unique_ids_raw_output___9:0' in input_names:
                onnx_inputs['unique_ids_raw_output___9:0'] = np.array([[0]], dtype=np.int64)
            
            # Run inference on-device
            outputs = self.session.run(None, onnx_inputs)
            
            # Get start and end logits
            start_logits = outputs[1] if len(outputs) > 1 else outputs[0]
            end_logits = outputs[0] if len(outputs) > 1 else outputs[0]
            
            # Find the best answer span by considering all valid combinations
            # Get top 20 start and end positions
            start_scores = np.argsort(start_logits[0])[::-1][:20]
            end_scores = np.argsort(end_logits[0])[::-1][:20]
            
            # Find the best valid span (where end >= start and within reasonable length)
            best_score = float('-inf')
            best_start = 0
            best_end = 0
            
            for start_idx in start_scores:
                for end_idx in end_scores:
                    # Valid span: end must be after start and within 30 tokens
                    if end_idx >= start_idx and (end_idx - start_idx) <= 30:
                        score = start_logits[0][start_idx] + end_logits[0][end_idx]
                        if score > best_score:
                            best_score = score
                            best_start = start_idx
                            best_end = end_idx
            
            # Extract answer tokens
            answer_tokens = input_ids[0][best_start:best_end + 1]
            
            # Decode the answer
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Clean up the answer
            answer = answer.strip()
            
            # Debug: Print what BERT found (optional - remove in production)
            print(f"DEBUG: Question: {question[:50]}...")
            print(f"DEBUG: Best answer span: [{best_start}:{best_end}]")
            print(f"DEBUG: Score: {best_score:.2f}")
            print(f"DEBUG: Answer: '{answer}'")
            
            # If answer is empty or just punctuation, try fallback
            if not answer or len(answer) < 2 or answer in ['.', ',', '?', '!']:
                print("DEBUG: Answer too short, using fallback")
                return self.fallback_answer(context, question)
            
            return f"{answer}\n\n✓ Answer extracted using BERT AI model (on-device)"
            
        except Exception as e:
            print(f"AI model error: {e}")
            import traceback
            traceback.print_exc()
            return self.fallback_answer(context, question)
    
    def fallback_answer(self, context, question):
        """Improved semantic answer extraction (fallback with better word matching)"""
        # Split context into sentences
        sentences = [s.strip() + '.' for s in context.replace('\n', ' ').split('.') if s.strip()]
        
        if not sentences:
            return "No answer found in the provided context."
        
        # Extract keywords from question with better normalization
        question_lower = question.lower().replace('?', '')
        question_words = set(question_lower.split())
        
        # Expanded stop words
        stop_words = {'what', 'where', 'when', 'who', 'why', 'how', 'is', 'are', 'was', 'were', 
                     'do', 'does', 'did', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                     'for', 'with', 'from', 'to', 'of', 'by', 'as', 'it', 'this', 'that', 'these',
                     'those', 'be', 'been', 'being', 'have', 'has', 'had'}
        
        keywords = question_words - stop_words
        
        # Add word stems and common variants for better matching
        expanded_keywords = set(keywords)
        for keyword in keywords:
            # Add common word variations
            if keyword.endswith('ing'):
                expanded_keywords.add(keyword[:-3])  # building -> build
                expanded_keywords.add(keyword[:-3] + 'ed')  # building -> builded
                expanded_keywords.add(keyword[:-3] + 't')  # building -> built
            elif keyword.endswith('ed'):
                expanded_keywords.add(keyword[:-2])  # created -> create
                expanded_keywords.add(keyword[:-2] + 'ing')  # created -> creating
                expanded_keywords.add(keyword[:-1])  # established -> establish
            elif keyword.endswith('s'):
                expanded_keywords.add(keyword[:-1])  # builds -> build
            
            # Add the base word too
            if len(keyword) > 3:
                expanded_keywords.add(keyword)
        
        if not expanded_keywords:
            # Return first sentence if no keywords
            return sentences[0] + "\n\n⚠ Basic answer (AI model provides better results)"
        
        # Score sentences with semantic matching
        best_score = 0
        best_sentence = sentences[0]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(sentence_lower.split())
            
            # Count exact matches
            exact_matches = sum(1 for kw in keywords if kw in sentence_lower)
            
            # Count fuzzy matches (word stems, variants)
            fuzzy_matches = sum(1 for kw in expanded_keywords 
                              if any(kw in word or word in kw for word in sentence_words) 
                              and kw not in keywords)
            
            # Calculate combined score (exact matches worth more)
            score = (exact_matches * 2) + fuzzy_matches
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        if best_score == 0:
            return "No relevant answer found in the provided context.\n\n⚠ AI model provides better semantic understanding"
        
        return f"{best_sentence}\n\n⚠ Basic semantic matching (AI model provides better accuracy)"
    
    def clear_text(self):
        """Clear all text fields"""
        self.context_box.delete('1.0', tk.END)
        self.question_box.delete('1.0', tk.END)
        self.result_box.delete('1.0', tk.END)

def main():
    """Main application entry point"""
    print("\n" + "=" * 70)
    print("  On-Device AI Reading Assistant")
    print("  Foundry Local + Windows AI Foundry - On-Device Processing")
    print("=" * 70)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Ask questions about your documents")
    print("  All processing happens locally on your device")
    print("=" * 70 + "\n")
    
    root = tk.Tk()
    app = PrivacyFirstAIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()