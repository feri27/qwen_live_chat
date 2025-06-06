from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, disconnect
from flask_cors import CORS
import asyncio
import json
import base64
import io
import wave
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime
import uuid
import torch
from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor
from transformers.utils.hub import cached_file
from gptqmodel import GPTQModel
from gptqmodel.models.base import BaseGPTQModel
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models._const import CPU, SUPPORTED_MODELS
from huggingface_hub import snapshot_download
from qwen_omni_utils import process_mm_info
import librosa
import tempfile
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import functools

# GPTQ Model Class for Qwen2.5-Omni
class Qwen25OmniThinkerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniForConditionalGeneration
    base_modules = [
        "thinker.model.embed_tokens", 
        "thinker.model.norm", 
        "token2wav", 
        "thinker.audio_tower", 
        "thinker.model.rotary_emb",
        "thinker.visual", 
        "talker"
    ]
    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
   
    def pre_quantize_generate_hook_start(self):
        from gptqmodel.models._const import CPU
        self.thinker.visual = self._move_to(self.thinker.visual, device=self.quantize_config.device)
        self.thinker.audio_tower = self._move_to(self.thinker.audio_tower, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        from gptqmodel.models._const import CPU
        self.thinker.visual = self._move_to(self.thinker.visual, device=CPU)
        self.thinker.audio_tower = self._move_to(self.thinker.audio_tower, device=CPU)

    def _move_to(self, module, device):
        """Helper function to move module to device"""
        if hasattr(module, 'to'):
            return module.to(device)
        return module

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

# Register the GPTQ model
MODEL_MAP["qwen2_5_omni"] = Qwen25OmniThinkerGPTQ
SUPPORTED_MODELS.extend(["qwen2_5_omni"])

@classmethod
def patched_from_config(cls, config, *args, **kwargs):
    kwargs.pop("trust_remote_code", None)
    model = cls._from_config(config, **kwargs)
    
    # Load speaker dictionary
    try:
        spk_path = cached_file(
            kwargs.get('model_path', 'Qwen/Qwen2.5-Omni-7B-GPTQ-Int4'),
            "spk_dict.pt",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if spk_path and hasattr(model, 'load_speakers'):
            model.load_speakers(spk_path)
    except Exception as e:
        logger.warning(f"Could not load speaker dictionary: {e}")
    
    return model

# Patch the from_config method
Qwen2_5OmniForConditionalGeneration.from_config = patched_from_config

# Qwen Omni integration with GPTQ
class QwenOmniClient:
    def __init__(self, model_path="Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device_map = {
            "thinker.model": "cuda" if torch.cuda.is_available() else "cpu", 
            "thinker.lm_head": "cuda" if torch.cuda.is_available() else "cpu", 
            "thinker.visual": "cpu",  # Keep visual on CPU to save VRAM
            "thinker.audio_tower": "cpu",  # Keep audio on CPU to save VRAM
            "talker": "cuda" if torch.cuda.is_available() else "cpu",  
            "token2wav": "cuda" if torch.cuda.is_available() else "cpu",  
        }
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the Qwen Omni GPTQ model and processor"""
        try:
            logger.info("Downloading and loading Qwen Omni GPTQ model...")
            
            # Download model if needed
            model_path = snapshot_download(repo_id=self.model_path)
            
            # Load GPTQ model with optimized device mapping
            self.model = GPTQModel.load(
                model_path, 
                device_map=self.device_map, 
                torch_dtype=torch.float16,   
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
            )
            
            # Load processor
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
            
            if torch.cuda.is_available():
                logger.info("GPTQ Model loaded with optimized GPU/CPU split")
                logger.info(f"Device map: {self.device_map}")
            else:
                logger.info("GPTQ Model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Error loading Qwen Omni GPTQ model: {e}")
            raise e
    
    def _save_audio_to_temp_file(self, audio_data: bytes) -> str:
        """Save audio bytes to a temporary file and return the path"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            return temp_file.name
    
    def _preprocess_audio(self, audio_data: bytes) -> np.ndarray:
        """Preprocess audio data to the required format"""
        try:
            # Save audio data to temporary file
            temp_path = self._save_audio_to_temp_file(audio_data)
            
            # Load audio using librosa
            audio, sr = librosa.load(temp_path, sr=16000)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return audio
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise e
    
    def speech_to_text(self, audio_data: bytes) -> str:
        """
        Convert speech to text using Qwen Omni
        """
        try:
            # Preprocess audio
            audio = self._preprocess_audio(audio_data)
            
            # Create messages for speech-to-text
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a speech recognition model."}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "Transcribe the audio into text."},
                ]},
            ]
            
            # Run inference
            result = self._run_inference(messages, audio_array=audio)
            return result
            
        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}")
            return f"Error transcribing audio: {str(e)}"
    
    def generate_response(self, text: str, conversation_history: List[Dict]) -> str:
        """
        Generate response using Qwen Omni LLM
        """
        try:
            # Create system message
            system_prompt = "You are Qwen, a helpful AI assistant created by Alibaba Cloud. You are knowledgeable, helpful, and honest."
            
            # Build conversation messages
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            ]
            
            # Add conversation history (last 10 messages to avoid context overflow)
            recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
            
            for msg in recent_history[:-1]:  # Exclude the last message as it's the current input
                messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}]
                })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": text}]
            })
            
            # Run inference
            result = self._run_inference(messages)
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _run_inference(self, messages: List[Dict], audio_array: np.ndarray = None) -> str:
        """
        Run inference with the Qwen Omni GPTQ model
        """
        try:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process multimedia information
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            
            # If we have audio_array, use it instead
            if audio_array is not None:
                audios = [audio_array] if audios is None or len(audios) == 0 else audios
            
            # Process inputs
            inputs = self.processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=True
            )
            
            # Move inputs to appropriate device (CUDA if available)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inputs = inputs.to(device).to(self.model.dtype)
            
            # Generate response with GPTQ model
            with torch.no_grad():
                output = self.model.generate(
                    **inputs, 
                    use_audio_in_video=True, 
                    return_audio=False,  # Disable audio return for now to save memory
                    thinker_max_new_tokens=256, 
                    thinker_do_sample=False,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode output
            response = self.processor.batch_decode(
                output, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # Extract the response (remove the input part)
            full_response = response[0]
            # Find the assistant's response part
            if "<|im_start|>assistant\n" in full_response:
                assistant_response = full_response.split("<|im_start|>assistant\n")[-1]
                if "<|im_end|>" in assistant_response:
                    assistant_response = assistant_response.split("<|im_end|>")[0]
                return assistant_response.strip()
            else:
                # Fallback: return the last part after the last user message
                parts = full_response.split("user\n")
                if len(parts) > 1:
                    return parts[-1].strip()
                return full_response.strip()
            
        except Exception as e:
            logger.error(f"Error in _run_inference: {e}")
            raise e
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                "reserved": torch.cuda.memory_reserved() / 1024 / 1024,    # MB
                "max_allocated": torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            }
        return {"message": "CUDA not available"}
    
    def clear_memory_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize SocketIO with CORS
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000", async_mode='threading')

# Enable CORS for regular HTTP routes
CORS(app, origins=["http://localhost:3000"])

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for running async operations
executor = ThreadPoolExecutor(max_workers=4)

# Initialize Qwen Omni client (this may take some time)
logger.info("Initializing Qwen Omni client...")
try:
    qwen_client = QwenOmniClient()
    logger.info("Qwen Omni client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qwen Omni client: {e}")
    qwen_client = None

# Store active connections and conversation history
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, str] = {}  # client_id -> session_id
        self.conversation_history: Dict[str, List[Dict]] = {}
    
    def connect(self, client_id: str, session_id: str):
        self.active_connections[client_id] = session_id
        if client_id not in self.conversation_history:
            self.conversation_history[client_id] = []
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    def is_connected(self, client_id: str) -> bool:
        return client_id in self.active_connections

manager = ConnectionManager()

def process_audio_data(audio_base64: str) -> bytes:
    """Convert base64 audio data to bytes"""
    try:
        # Remove data URL prefix if present
        if "data:audio" in audio_base64:
            audio_base64 = audio_base64.split(",")[1]
        
        audio_data = base64.b64decode(audio_base64)
        return audio_data
    except Exception as e:
        logger.error(f"Error processing audio data: {e}")
        raise ValueError("Invalid audio data")

def run_in_executor(func):
    """Decorator to run synchronous functions in thread pool"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return executor.submit(func, *args, **kwargs).result()
    return wrapper

# HTTP Routes
@app.route('/')
def root():
    return jsonify({
        "message": "Qwen Omni AI Assistant Backend", 
        "status": "running",
        "model_loaded": qwen_client is not None
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if qwen_client is not None else "not_loaded",
        "gpu_available": torch.cuda.is_available()
    })

@app.route('/conversation/<client_id>')
def get_conversation_history(client_id):
    """Get conversation history for a client"""
    if client_id in manager.conversation_history:
        return jsonify({"history": manager.conversation_history[client_id]})
    return jsonify({"history": []})

@app.route('/conversation/<client_id>', methods=['DELETE'])
def clear_conversation_history(client_id):
    """Clear conversation history for a client"""
    if client_id in manager.conversation_history:
        manager.conversation_history[client_id] = []
        return jsonify({"message": "Conversation history cleared"})
    return jsonify({"message": "Client not found"})

@app.route('/memory')
def get_memory_usage():
    """Get current GPU memory usage"""
    if qwen_client:
        return jsonify(qwen_client.get_memory_usage())
    return jsonify({"error": "Model not loaded"})

@app.route('/memory/clear', methods=['POST'])
def clear_memory_cache():
    """Clear GPU memory cache"""
    if qwen_client:
        qwen_client.clear_memory_cache()
        return jsonify({"message": "Memory cache cleared", "usage": qwen_client.get_memory_usage()})
    return jsonify({"error": "Model not loaded"})

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    client_id = request.args.get('client_id')
    if not client_id:
        disconnect()
        return False
    
    if qwen_client is None:
        emit('error', {'message': 'Model not loaded'})
        disconnect()
        return False
    
    session_id = request.sid
    manager.connect(client_id, session_id)
    logger.info(f"Client {client_id} connected with session {session_id}")
    
    emit('connected', {'message': f'Connected as {client_id}'})

@socketio.on('disconnect')
def handle_disconnect():
    client_id = None
    # Find client_id by session_id
    for cid, sid in manager.active_connections.items():
        if sid == request.sid:
            client_id = cid
            break
    
    if client_id:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")

def process_audio_async(client_id, audio_data):
    """Process audio in a separate thread"""
    try:
        # Send processing status
        socketio.emit('status', {
            'message': 'Processing audio...',
            'timestamp': datetime.now().isoformat()
        }, room=manager.active_connections[client_id])
        
        # Convert speech to text
        transcribed_text = qwen_client.speech_to_text(audio_data)
        
        # Send transcription
        socketio.emit('transcription', {
            'text': transcribed_text,
            'timestamp': datetime.now().isoformat()
        }, room=manager.active_connections[client_id])
        
        # Add to conversation history
        manager.conversation_history[client_id].append({
            "role": "user",
            "content": transcribed_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate AI response
        ai_response = qwen_client.generate_response(
            transcribed_text, 
            manager.conversation_history[client_id]
        )
        
        # Add AI response to history
        manager.conversation_history[client_id].append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send AI response
        socketio.emit('response', {
            'text': ai_response,
            'timestamp': datetime.now().isoformat()
        }, room=manager.active_connections[client_id])
        
    except Exception as e:
        logger.error(f"Error processing audio for client {client_id}: {e}")
        socketio.emit('error', {
            'message': f'Error processing audio: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }, room=manager.active_connections[client_id])

def process_text_async(client_id, user_text):
    """Process text in a separate thread"""
    try:
        # Add to conversation history
        manager.conversation_history[client_id].append({
            "role": "user",
            "content": user_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate AI response
        ai_response = qwen_client.generate_response(
            user_text, 
            manager.conversation_history[client_id]
        )
        
        # Add AI response to history
        manager.conversation_history[client_id].append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send AI response
        socketio.emit('response', {
            'text': ai_response,
            'timestamp': datetime.now().isoformat()
        }, room=manager.active_connections[client_id])
        
    except Exception as e:
        logger.error(f"Error processing text for client {client_id}: {e}")
        socketio.emit('error', {
            'message': f'Error processing text: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }, room=manager.active_connections[client_id])

@socketio.on('audio')
def handle_audio(data):
    client_id = data.get('client_id')
    audio_base64 = data.get('audio')
    
    if not client_id or not manager.is_connected(client_id):
        emit('error', {'message': 'Client not connected'})
        return
    
    try:
        audio_data = process_audio_data(audio_base64)
        # Process audio in background thread
        threading.Thread(
            target=process_audio_async, 
            args=(client_id, audio_data)
        ).start()
        
    except Exception as e:
        logger.error(f"Error handling audio for client {client_id}: {e}")
        emit('error', {
            'message': f'Error processing audio: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })

@socketio.on('text')
def handle_text(data):
    client_id = data.get('client_id')
    user_text = data.get('text')
    
    if not client_id or not manager.is_connected(client_id):
        emit('error', {'message': 'Client not connected'})
        return
    
    if not user_text:
        emit('error', {'message': 'No text provided'})
        return
    
    try:
        # Process text in background thread
        threading.Thread(
            target=process_text_async, 
            args=(client_id, user_text)
        ).start()
        
    except Exception as e:
        logger.error(f"Error handling text for client {client_id}: {e}")
        emit('error', {
            'message': f'Error processing text: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })

@socketio.on('ping')
def handle_ping():
    emit('pong')

if __name__ == '__main__':
    # Run the Flask-SocketIO app
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=8000, 
        debug=False,
        use_reloader=False,  # Disable reloader to prevent model loading twice
        log_output=True
    )