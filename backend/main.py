from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import librosa
import tempfile
import os

# Qwen Omni integration
class QwenOmniClient:
    def __init__(self, model_path="Qwen/Qwen2.5-Omni-7B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the Qwen Omni model and processor"""
        try:
            logger.info("Loading Qwen Omni model...")
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager"
            )
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Error loading Qwen Omni model: {e}")
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
    
    async def speech_to_text(self, audio_data: bytes) -> str:
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
            result = await self._run_inference(messages, audio_array=audio)
            return result
            
        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}")
            return f"Error transcribing audio: {str(e)}"
    
    async def generate_response(self, text: str, conversation_history: List[Dict]) -> str:
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
            result = await self._run_inference(messages)
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    async def _run_inference(self, messages: List[Dict], audio_array: np.ndarray = None) -> str:
        """
        Run inference with the Qwen Omni model
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
            
            # Move inputs to model device
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs, 
                    use_audio_in_video=True, 
                    return_audio=False, 
                    thinker_max_new_tokens=256, 
                    thinker_do_sample=False,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
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
                return full_response.split("user\n")[-1].strip()
            
        except Exception as e:
            logger.error(f"Error in _run_inference: {e}")
            raise e

# Initialize FastAPI app
app = FastAPI(title="Qwen Omni AI Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_history: Dict[str, List[Dict]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.conversation_history[client_id] = []
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.conversation_history:
            del self.conversation_history[client_id]
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_text(json.dumps(message))

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
        raise HTTPException(status_code=400, detail="Invalid audio data")

@app.get("/")
async def root():
    return {
        "message": "Qwen Omni AI Assistant Backend", 
        "status": "running",
        "model_loaded": qwen_client is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if qwen_client is not None else "not_loaded",
        "gpu_available": torch.cuda.is_available()
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    if qwen_client is None:
        await websocket.close(code=1000, reason="Model not loaded")
        return
        
    await manager.connect(websocket, client_id)
    logger.info(f"Client {client_id} connected")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio":
                # Process audio data
                try:
                    audio_data = process_audio_data(message["audio"])
                    
                    # Send processing status
                    await manager.send_message(client_id, {
                        "type": "status",
                        "message": "Processing audio...",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Convert speech to text
                    transcribed_text = await qwen_client.speech_to_text(audio_data)
                    
                    # Send transcription
                    await manager.send_message(client_id, {
                        "type": "transcription",
                        "text": transcribed_text,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Add to conversation history
                    manager.conversation_history[client_id].append({
                        "role": "user",
                        "content": transcribed_text,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Generate AI response
                    ai_response = await qwen_client.generate_response(
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
                    await manager.send_message(client_id, {
                        "type": "response",
                        "text": ai_response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing audio for client {client_id}: {e}")
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": f"Error processing audio: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif message["type"] == "text":
                # Handle text input
                try:
                    user_text = message["text"]
                    
                    # Add to conversation history
                    manager.conversation_history[client_id].append({
                        "role": "user",
                        "content": user_text,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Generate AI response
                    ai_response = await qwen_client.generate_response(
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
                    await manager.send_message(client_id, {
                        "type": "response",
                        "text": ai_response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing text for client {client_id}: {e}")
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": f"Error processing text: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)

@app.get("/conversation/{client_id}")
async def get_conversation_history(client_id: str):
    """Get conversation history for a client"""
    if client_id in manager.conversation_history:
        return {"history": manager.conversation_history[client_id]}
    return {"history": []}

@app.delete("/conversation/{client_id}")
async def clear_conversation_history(client_id: str):
    """Clear conversation history for a client"""
    if client_id in manager.conversation_history:
        manager.conversation_history[client_id] = []
        return {"message": "Conversation history cleared"}
    return {"message": "Client not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")