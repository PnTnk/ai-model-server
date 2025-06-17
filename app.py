# app.py (Dynamic Loading - Cleaned & Optimized Version)

import os
import sys
import time
import logging
import threading
import re
import gc
from typing import Dict, Any, Optional, List
import base64
import io

import torch
import numpy as np
from flask import Flask, request, jsonify

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForVisualQuestionAnswering,
    BitsAndBytesConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor
)

from PIL import Image
import librosa
import soundfile as sf
from pdf2image import convert_from_bytes

# Import optional dependencies with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Import utility modules
from utils import tmp_manage


# ==================== CONFIGURATION ====================

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

MODELS_CONFIG = {
    "deepseek": {
        "path": "./models/deepseek-8b", 
        "loader": AutoModelForCausalLM, 
        "processor_loader": AutoTokenizer, 
        "type": "text", 
        "quant_bits": 4, 
        "max_memory": {0: "8GiB", "cpu": "32GiB"}
    },
    "qwen_vl": {
        "path": "./models/qwen-vl-3b", 
        "loader": AutoModelForImageTextToText, 
        "processor_loader": AutoProcessor, 
        "type": "vision", 
        "quant_bits": 4, 
        "max_memory": {0: "4GiB", "cpu": "16GiB"}
    },
    "qwen_audio": {
        "path": "./models/qwen-audio-7b", 
        "loader": AutoModelForSeq2SeqLM, 
        "processor_loader": AutoProcessor, 
        "type": "audio", 
        "quant_bits": 4, 
        "max_memory": {0: "8GiB", "cpu": "32GiB"}
    },
    "blip_vqa": {
        "path": "./models/blip-vqa-base", 
        "loader": AutoModelForVisualQuestionAnswering, 
        "processor_loader": AutoProcessor, 
        "type": "vqa", 
        "quant_bits": 8
    },
    "qwen_omni": {
        "path": "./models/qwen-omni-3b",
        "loader": Qwen2_5OmniForConditionalGeneration,
        "processor_loader": Qwen2_5OmniProcessor,
        "type": "omni",
        "quant_bits": 4,
        "max_memory": {0: "4GiB", "cpu": "16GiB"}
    }
}

model_lock = threading.Lock()


# ==================== UTILITY FUNCTIONS ====================

def get_quantization_config(quant_bits: Optional[int]) -> Optional[BitsAndBytesConfig]:
    """Create quantization configuration based on bits specified."""
    if quant_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif quant_bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True, 
            llm_int8_enable_fp32_cpu_offload=True
        )
    return None


def cleanup_gpu_memory():
    """Clean up GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def unload_resources(components_dict: Dict[str, Any]):
    """Unload model components and free memory."""
    if not components_dict:
        return
    
    keys_to_delete = list(components_dict.keys())
    for key in keys_to_delete:
        del components_dict[key]
    
    cleanup_gpu_memory()


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def resize_image_for_memory(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize image to save memory while maintaining aspect ratio."""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        logging.info(f"Resized image to {new_size} for memory efficiency")
    return image


def save_temp_file(data: bytes, filename: str) -> str:
    """Save binary data to temporary file and return path."""
    os.makedirs("./tmp", exist_ok=True)
    temp_path = f"./tmp/{filename}"
    with open(temp_path, "wb") as f:
        f.write(data)
    return temp_path


def extract_video_frames(video_path: str, max_frames: int = 8) -> List[Image.Image]:
    """Extract frames from video file using OpenCV."""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV not available for video processing")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logging.info(f"Video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
        
        # Sample frames evenly
        if total_frames > max_frames:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            frame_indices = list(range(total_frames))
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert BGR to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = resize_image_for_memory(frame_pil, max_size=512)
            frames.append(frame_pil)
        
        logging.info(f"Extracted {len(frames)} frames successfully")
        return frames
        
    finally:
        cap.release()


# ==================== MODEL MANAGEMENT ====================

def manage_model(model_name: str) -> Optional[Dict[str, Any]]:
    """Load model and processor dynamically."""
    if model_name not in MODELS_CONFIG:
        logging.error(f"Configuration for model '{model_name}' not found.")
        return None
    
    config = MODELS_CONFIG[model_name]
    model_path = config["path"]
    
    logging.info(f"Loading model: '{model_name}' from '{model_path}'")
    
    if not os.path.isdir(model_path):
        logging.error(f"Directory for model '{model_name}' not found.")
        return None

    try:
        quantization_config = get_quantization_config(config.get("quant_bits"))
        
        model = config["loader"].from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
            max_memory=config.get("max_memory")
        )
        
        processor = config["processor_loader"].from_pretrained(
            model_path, 
            trust_remote_code=True,
            use_fast=True
        )
        
        return {"model": model, "processor": processor}
        
    except Exception as e:
        logging.error(f"Failed to load model '{model_name}': {e}", exc_info=True)
        return None


# ==================== TEXT GENERATION ====================

@app.route('/generate/deepseek', methods=['POST'])
def generate_deepseek_text():
    """Generate text using DeepSeek model."""
    with model_lock:
        components = None
        try:
            components = manage_model("deepseek")
            if components is None:
                return jsonify({"error": "Could not load DeepSeek model."}), 503
            
            data = request.get_json()
            if not data or 'prompt' not in data:
                return jsonify({"error": "'prompt' key is required."}), 400
            
            prompt_text = data['prompt']
            model = components["model"]
            tokenizer = components["processor"]
            
            logging.info(f"DeepSeek generation: {prompt_text[:50]}...")
            
            messages = [{"role": "user", "content": prompt_text}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=data.get('max_new_tokens', 256),
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            input_token_length = inputs.input_ids.shape[-1]
            response_text = tokenizer.decode(
                outputs[0][input_token_length:], 
                skip_special_tokens=True
            )
            
            # Remove thinking tags
            think_pattern = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
            cleaned_text = think_pattern.sub('', response_text).strip()
            
            return jsonify({"response": cleaned_text})
            
        except Exception as e:
            logging.error(f"DeepSeek generation error: {e}", exc_info=True)
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500
        finally:
            if components:
                unload_resources(components)


# ==================== VISION GENERATION ====================

@app.route('/generate/qwen_vl', methods=['POST'])
def generate_qwen_vl_vision_text():
    """Generate text from image using Qwen-VL model."""
    with model_lock:
        components = None
        try:
            components = manage_model("qwen_vl")
            if components is None:
                return jsonify({"error": "Could not load Qwen-VL model."}), 503
            
            data = request.get_json()
            if not data or 'prompt' not in data or 'image' not in data:
                return jsonify({"error": "'prompt' and 'image' keys are required."}), 400
            
            prompt_text = data['prompt']
            model = components["model"]
            processor = components["processor"]
            
            image = decode_base64_image(data['image'])
            
            logging.info(f"Qwen-VL generation: {prompt_text[:50]}...")
            
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": prompt_text}
                ]
            }]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs, 
                    max_new_tokens=data.get('max_new_tokens', 512)
                )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            return jsonify({"response": response_text.strip()})
            
        except Exception as e:
            logging.error(f"Qwen-VL generation error: {e}", exc_info=True)
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500
        finally:
            if components:
                unload_resources(components)


@app.route('/generate/blip_vqa', methods=['POST'])
def generate_blip_vqa():
    """Generate answer using BLIP-VQA model."""
    with model_lock:
        components = None
        try:
            components = manage_model("blip_vqa")
            if components is None:
                return jsonify({"error": "Could not load BLIP-VQA model."}), 503
            
            data = request.get_json()
            if not data or 'prompt' not in data or 'image' not in data:
                return jsonify({"error": "'prompt' and 'image' keys are required."}), 400
            
            question = data['prompt']
            model = components["model"]
            processor = components["processor"]
            
            image = decode_base64_image(data['image'])
            
            logging.info(f"BLIP-VQA question: {question[:50]}...")
            
            inputs = processor(
                images=image, text=question, return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_length=50)
            
            answer = processor.decode(generated_ids[0], skip_special_tokens=True)
            return jsonify({"response": answer.strip()})
            
        except Exception as e:
            logging.error(f"BLIP-VQA generation error: {e}", exc_info=True)
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500
        finally:
            if components:
                unload_resources(components)


# ==================== AUDIO GENERATION ====================

@app.route('/generate/qwen_audio', methods=['POST'])
def generate_qwen_audio():
    """Generate text from audio using Qwen-Audio model."""
    with model_lock:
        components = None
        try:
            components = manage_model("qwen_audio")
            if components is None:
                return jsonify({"error": "Could not load Qwen-Audio model."}), 503
            
            data = request.get_json()
            if not data or 'prompt' not in data or 'audio' not in data:
                return jsonify({"error": "'prompt' and 'audio' keys are required."}), 400
            
            prompt_text = data['prompt']
            model = components["model"]
            processor = components["processor"]
            
            media_bytes = base64.b64decode(data['audio'])
            sampling_rate = processor.feature_extractor.sampling_rate
            audio_input, _ = librosa.load(
                io.BytesIO(media_bytes), sr=sampling_rate, mono=True
            )
            
            logging.info(f"Qwen-Audio generation: {prompt_text[:50]}...")
            
            text_with_placeholder = (
                f"<|im_start|>user\n<|AUDIO|>{prompt_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            model_inputs = processor(
                text=text_with_placeholder,
                audio=audio_input,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs, 
                    max_new_tokens=data.get('max_new_tokens', 512)
                )
            
            response_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[-1]
            
            return jsonify({"response": response_text.strip()})
            
        except Exception as e:
            logging.error(f"Qwen-Audio generation error: {e}", exc_info=True)
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500
        finally:
            if components:
                unload_resources(components)


# ==================== OMNI GENERATION ====================

@app.route('/generate/qwen_omni', methods=['POST'])
def generate_qwen_omni():
    """Generate using Qwen-Omni model with multimodal support."""
    with model_lock:
        components = None
        temp_files_to_clean = []
        
        try:
            components = manage_model("qwen_omni")
            if components is None:
                return jsonify({"error": "Could not load Qwen-Omni model."}), 503

            data = request.get_json()
            if not data or 'prompt' not in data:
                return jsonify({"error": "'prompt' key is required."}), 400
            
            prompt_text = data['prompt']
            model = components["model"]
            processor = components["processor"]
            
            logging.info(f"Qwen-Omni generation: {prompt_text[:50]}...")
            
            # Memory optimization
            cleanup_gpu_memory()
            
            # Disable talker for memory efficiency
            try:
                if hasattr(model, 'disable_talker'):
                    model.disable_talker()
            except:
                pass

            # Check multimodal inputs
            has_image = bool('image' in data and data['image'])
            has_audio = bool('audio' in data and data['audio'])
            has_video = bool('video' in data and data['video'])
            
            multimodal_count = sum([has_image, has_audio, has_video])
            logging.info(f"Multimodal inputs - Image: {has_image}, Audio: {has_audio}, Video: {has_video}")
            
            # Memory safety check for multiple modalities
            if multimodal_count >= 2:
                logging.warning(f"Multiple modalities detected ({multimodal_count}) - using CPU fallback")
                return generate_qwen_omni_cpu_fallback(data, components, temp_files_to_clean)
            
            # Single modality processing
            return generate_qwen_omni_single_modal(data, components, temp_files_to_clean)

        except torch.cuda.OutOfMemoryError as oom_error:
            logging.error(f"CUDA OOM: {oom_error}")
            cleanup_gpu_memory()
            return generate_qwen_omni_cpu_fallback(data, components, temp_files_to_clean)
            
        except Exception as e:
            logging.error(f"Qwen-Omni generation error: {e}", exc_info=True)
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500

        finally:
            # Cleanup temp files
            for temp_file in temp_files_to_clean:
                if isinstance(temp_file, str) and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                        
            if components:
                unload_resources(components)


def generate_qwen_omni_single_modal(data: Dict[str, Any], components: Dict[str, Any], temp_files_to_clean: List[str]):
    """Process single modality input for Qwen-Omni."""
    model = components["model"]
    processor = components["processor"]
    prompt_text = data['prompt']
    
    # Create conversation
    conversation = [
        {
            "role": "system",
            "content": [{
                "type": "text", 
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            }]
        },
        {
            "role": "user", 
            "content": [{"type": "text", "text": prompt_text}]
        }
    ]
    
    # Process different modalities
    if 'image' in data and data['image']:
        try:
            image = decode_base64_image(data['image'])
            image = resize_image_for_memory(image, max_size=512)
            conversation[1]["content"].insert(0, {"type": "image"})
            
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                inputs = processor(
                    text=text[0] if isinstance(text, list) else text,
                    images=[image],
                    return_tensors="pt"
                ).to(model.device)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=min(data.get('max_new_tokens', 128), 128),
                        do_sample=True,
                        temperature=0.7,
                        use_cache=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                    
        except Exception as e:
            logging.error(f"Image processing error: {e}")
            return jsonify({"error": "Image processing failed"}), 400
            
    elif 'audio' in data and data['audio']:
        try:
            audio_bytes = base64.b64decode(data['audio'])
            temp_audio_path = save_temp_file(audio_bytes, f"temp_audio_{os.getpid()}.wav")
            temp_files_to_clean.append(temp_audio_path)
            
            conversation[1]["content"].insert(0, {"type": "audio", "audio": temp_audio_path})
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            
            # Load audio with reduced quality for memory efficiency
            audio_data, sr = librosa.load(temp_audio_path, sr=8000, mono=True)
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                inputs = processor(
                    text=text[0] if isinstance(text, list) else text,
                    audio=audio_data,
                    return_tensors="pt"
                ).to(model.device)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=min(data.get('max_new_tokens', 128), 128),
                        do_sample=True,
                        temperature=0.7,
                        use_cache=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                    
        except Exception as e:
            logging.error(f"Audio processing error: {e}")
            return jsonify({"error": "Audio processing failed"}), 400
        
    elif 'video' in data and data['video']:
        try:
            if not CV2_AVAILABLE:
                return jsonify({
                    "error": "Video processing requires opencv-python. Install with: pip install opencv-python"
                }), 400
            
            video_bytes = base64.b64decode(data['video'])
            temp_video_path = save_temp_file(video_bytes, f"temp_video_{os.getpid()}.mp4")
            temp_files_to_clean.append(temp_video_path)
            
            frames = extract_video_frames(temp_video_path, max_frames=8)
            
            if not frames:
                raise ValueError("Could not extract any frames from video")
            
            conversation[1]["content"].insert(0, {"type": "video"})
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                try:
                    # Try videos parameter first
                    inputs = processor(
                        text=text[0] if isinstance(text, list) else text,
                        videos=frames,
                        return_tensors="pt"
                    ).to(model.device)
                    
                except Exception:
                    # Fallback to images parameter
                    inputs = processor(
                        text=text[0] if isinstance(text, list) else text,
                        images=frames,
                        return_tensors="pt"
                    ).to(model.device)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=min(data.get('max_new_tokens', 128), 128),
                        do_sample=True,
                        temperature=0.7,
                        use_cache=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                    
        except Exception as e:
            logging.error(f"Video processing error: {e}")
            # Text-only fallback for video errors
            return generate_text_only_response(data, components, "video processing failed")
                
    else:
        # Text-only mode
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        inputs = processor.tokenizer(
            text[0] if isinstance(text, list) else text,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=data.get('max_new_tokens', 256),
                do_sample=True,
                temperature=data.get('temperature', 0.7),
                pad_token_id=processor.tokenizer.eos_token_id
            )
    
    # Decode response
    try:
        if hasattr(inputs, 'input_ids'):
            input_length = inputs.input_ids.shape[-1]
            response_tokens = generated_ids[0][input_length:]
        else:
            response_tokens = generated_ids[0]
            
        response_text = processor.tokenizer.decode(
            response_tokens,
            skip_special_tokens=True
        )
        
        return jsonify({
            "response": response_text.strip(),
            "model_type": "qwen_omni",
            "generation_method": "single_modal_gpu",
            "memory_optimized": True
        })
        
    except Exception as e:
        logging.error(f"Decode error: {e}")
        return jsonify({"error": "Response decoding failed"}), 500


def generate_qwen_omni_cpu_fallback(data: Dict[str, Any], components: Dict[str, Any], temp_files_to_clean: List[str]):
    """CPU fallback for multimodal processing."""
    model = components["model"]
    processor = components["processor"]
    prompt_text = data['prompt']
    
    logging.info("Using CPU fallback strategy (text-only mode)")
    
    try:
        cleanup_gpu_memory()
        
        # Create text-only conversation
        conversation = [
            {
                "role": "system",
                "content": [{
                    "type": "text", 
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                }]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}]
            }
        ]
        
        # Notify about multimedia content limitation
        if any(k in data for k in ['image', 'audio', 'video']):
            enhanced_prompt = (
                f"{prompt_text}\n\n[Note: I can see you've provided multimedia content "
                f"(image/audio/video), but I'm currently processing in text-only mode due to "
                f"GPU memory constraints. Please describe the multimedia content in your prompt "
                f"if you'd like me to respond to it specifically.]"
            )
            conversation[1]["content"][0]["text"] = enhanced_prompt
        
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        inputs = processor.tokenizer(
            text[0] if isinstance(text, list) else text,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=min(data.get('max_new_tokens', 128), 128),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                output_attentions=False,
                output_hidden_states=False,
                return_dict_in_generate=False
            )
        
        input_length = inputs.input_ids.shape[-1]
        response_tokens = generated_ids[0][input_length:]
        response_text = processor.tokenizer.decode(
            response_tokens,
            skip_special_tokens=True
        )
        
        # Cleanup memory after generation
        del inputs, generated_ids, response_tokens
        cleanup_gpu_memory()
        
        return jsonify({
            "response": response_text.strip(),
            "model_type": "qwen_omni", 
            "generation_method": "gpu_text_only_fallback",
            "memory_optimized": True,
            "note": "Processed in text-only mode due to GPU memory constraints."
        })
        
    except torch.cuda.OutOfMemoryError as oom_error:
        logging.error(f"Even text-only fallback failed with OOM: {oom_error}")
        
        return jsonify({
            "response": (
                f"I understand you want me to process your request: '{prompt_text[:100]}...'. "
                f"However, I'm currently experiencing memory constraints. Please try with a "
                f"shorter prompt or restart the service."
            ),
            "model_type": "qwen_omni",
            "generation_method": "emergency_fallback", 
            "memory_optimized": True,
            "error": "GPU memory exhausted"
        })
        
    except Exception as e:
        logging.error(f"CPU fallback error: {e}")
        
        return jsonify({
            "response": (
                f"I'm having technical difficulties processing your request. "
                f"The issue is: {str(e)[:100]}. Please try again with a simpler prompt."
            ),
            "model_type": "qwen_omni",
            "generation_method": "error_fallback",
            "error": str(e)
        }), 500


def generate_text_only_response(data: Dict[str, Any], components: Dict[str, Any], error_reason: str):
    """Generate text-only response as fallback."""
    try:
        model = components["model"]
        processor = components["processor"]
        prompt_text = data['prompt']
        
        enhanced_prompt = (
            f"{prompt_text}\n\n[Note: I received multimedia content but encountered "
            f"technical difficulties ({error_reason}). Please describe what happens in "
            f"the content so I can provide a helpful response.]"
        )
        
        conversation = [
            {
                "role": "system",
                "content": [{
                    "type": "text", 
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                }]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": enhanced_prompt}]
            }
        ]
        
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor.tokenizer(
            text[0] if isinstance(text, list) else text,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=min(data.get('max_new_tokens', 128), 128),
                do_sample=True,
                temperature=0.7,
                use_cache=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        if hasattr(inputs, 'input_ids'):
            input_length = inputs.input_ids.shape[-1]
            response_tokens = generated_ids[0][input_length:]
        else:
            response_tokens = generated_ids[0]
            
        response_text = processor.tokenizer.decode(
            response_tokens,
            skip_special_tokens=True
        )
        
        return jsonify({
            "response": response_text.strip(),
            "model_type": "qwen_omni",
            "generation_method": "fallback_text_only",
            "memory_optimized": True,
            "note": f"Multimedia processing failed ({error_reason}), responded in text-only mode"
        })
        
    except Exception as fallback_error:
        logging.error(f"Text-only fallback failed: {fallback_error}")
        return jsonify({"error": f"Complete processing failure: {str(fallback_error)}"}), 500


# ==================== ADMIN ENDPOINTS ====================

@app.route('/admin/memory_status', methods=['GET'])
def memory_status():
    """Check memory status."""
    status = {
        "timestamp": time.time(),
        "gpu_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        gpu_free = gpu_memory - gpu_reserved
        
        status.update({
            "gpu_total_gb": round(gpu_memory, 2),
            "gpu_allocated_gb": round(gpu_allocated, 2),
            "gpu_reserved_gb": round(gpu_reserved, 2),
            "gpu_free_gb": round(gpu_free, 2),
            "gpu_utilization_percent": round((gpu_allocated / gpu_memory) * 100, 1),
            "memory_warning": gpu_free < 1.0,
            "multimodal_safe": gpu_free >= 2.0
        })
    
    return jsonify(status)


@app.route('/admin/emergency_cleanup', methods=['POST'])
def emergency_cleanup():
    """Emergency memory cleanup."""
    try:
        cleanup_gpu_memory()
        
        # Remove all temp files
        temp_dir = "./tmp"
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
        logging.info("Emergency cleanup completed")
        
        return jsonify({
            "status": "success",
            "message": "Emergency cleanup completed",
            "timestamp": time.time()
        })
        
    except Exception as e:
        logging.error(f"Emergency cleanup failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/admin/temp_status', methods=['GET'])
def temp_status():
    """View temp files status."""
    return jsonify(tmp_manage.get_temp_status())


@app.route('/admin/cleanup_temp', methods=['POST'])
def cleanup_temp_files_endpoint():
    """Clean up all temp files."""
    return jsonify(tmp_manage.cleanup_all_temp_files())


@app.route('/admin/model_status', methods=['GET'])
def model_status():
    """Check status of all models."""
    status = {
        "timestamp": time.time(),
        "available_models": list(MODELS_CONFIG.keys()),
        "model_configs": {},
        "dependencies": {
            "opencv_available": CV2_AVAILABLE,
            "librosa_available": True
        }
    }
    
    for model_name, config in MODELS_CONFIG.items():
        model_path = config["path"]
        status["model_configs"][model_name] = {
            "type": config["type"],
            "path": model_path,
            "path_exists": os.path.isdir(model_path),
            "quant_bits": config.get("quant_bits"),
            "max_memory": config.get("max_memory", "auto")
        }
    
    return jsonify(status)


# ==================== HEALTH CHECK ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "available_endpoints": [
            "/generate/deepseek",
            "/generate/qwen_vl", 
            "/generate/qwen_audio",
            "/generate/blip_vqa",
            "/generate/qwen_omni",
            "/admin/memory_status",
            "/admin/model_status",
            "/admin/emergency_cleanup",
            "/admin/temp_status",
            "/admin/cleanup_temp",
            "/health"
        ],
        "dependencies": {
            "opencv": CV2_AVAILABLE,
            "librosa": True,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
    })


# ==================== MAIN APPLICATION ====================

if __name__ == '__main__':
    # Create temp directory if it doesn't exist
    os.makedirs("./tmp", exist_ok=True)
    
    logging.info("🚀 Starting AI Model API Server...")
    logging.info(f"📊 Available models: {list(MODELS_CONFIG.keys())}")
    logging.info(f"🔧 Dependencies - OpenCV: {CV2_AVAILABLE}, CUDA: {torch.cuda.is_available()}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)