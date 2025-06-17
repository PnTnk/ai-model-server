# download_model.py
import os
import logging
import time
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForVisualQuestionAnswering,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor
)
from huggingface_hub import snapshot_download, hf_hub_download
import torch

# --- การตั้งค่า ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- รายการโมเดลที่ต้องการดาวน์โหลด ---
MODELS_TO_DOWNLOAD = {
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": {
        "sub_dir": "deepseek-8b",
        "model_loader": AutoModelForCausalLM,
        "processor_loader": AutoTokenizer,
        "priority": 1,
        "description": "DeepSeek reasoning model (8B params)"
    },
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "sub_dir": "qwen-vl-3b",
        "model_loader": AutoModelForImageTextToText,
        "processor_loader": AutoProcessor,
        "priority": 2,
        "description": "Qwen vision-language model (3B params)"
    },
    "Qwen/Qwen2-Audio-7B-Instruct": {
        "sub_dir": "qwen-audio-7b",
        "model_loader": AutoModelForSeq2SeqLM,
        "processor_loader": AutoProcessor,
        "priority": 3,
        "description": "Qwen audio processing model (7B params)"
    },
    "Qwen/Qwen2.5-Omni-3B": {
        "sub_dir": "qwen-omni-3b",
        "model_loader": Qwen2_5OmniForConditionalGeneration,
        "processor_loader": Qwen2_5OmniProcessor,
        "download_method": "snapshot",
        "priority": 4,
        "description": "Qwen multimodal omni model (3B params)",
        "special_files": ["spk_dict.pt", "tokenizer.json"]  # ไฟล์พิเศษที่ต้องการ
    },
    "Salesforce/blip-vqa-base": {
        "sub_dir": "blip-vqa-base",
        "model_loader": AutoModelForVisualQuestionAnswering,
        "processor_loader": AutoProcessor,
        "priority": 5,
        "description": "BLIP visual question answering model"
    },
}

# โฟลเดอร์หลักสำหรับเก็บโมเดลทั้งหมด
BASE_SAVE_DIRECTORY = "./models"

def check_system_resources():
    """ตรวจสอบทรัพยากรระบบก่อนดาวน์โหลด"""
    try:
        import psutil
        
        # ตรวจสอบ RAM
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # ตรวจสอบพื้นที่ดิสก์
        disk = psutil.disk_usage('.')
        free_gb = disk.free / (1024**3)
        
        logging.info(f"System resources:")
        logging.info(f"  Available RAM: {available_gb:.1f} GB")
        logging.info(f"  Free disk space: {free_gb:.1f} GB")
        
        if available_gb < 8:
            logging.warning("Low RAM detected. Consider closing other applications.")
        
        if free_gb < 50:
            logging.warning("Low disk space detected. Models require ~40-50GB total.")
            
    except ImportError:
        logging.info("Install 'psutil' for system resource monitoring: pip install psutil")

def download_special_files(model_name, save_path, special_files):
    """ดาวน์โหลดไฟล์พิเศษที่จำเป็น"""
    for file_name in special_files:
        file_path = os.path.join(save_path, file_name)
        if not os.path.exists(file_path):
            try:
                logging.info(f"Downloading special file: {file_name}")
                hf_hub_download(
                    repo_id=model_name,
                    filename=file_name,
                    local_dir=save_path,
                    local_dir_use_symlinks=False
                )
                logging.info(f"Successfully downloaded: {file_name}")
            except Exception as e:
                logging.warning(f"Could not download {file_name}: {e}")
                logging.info(f"Model may work without {file_name} (limited functionality)")

def estimate_download_size(model_name):
    """ประมาณขนาดไฟล์ที่ต้องดาวน์โหลด"""
    size_estimates = {
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "16 GB",
        "Qwen/Qwen2.5-VL-3B-Instruct": "6 GB", 
        "Qwen/Qwen2-Audio-7B-Instruct": "14 GB",
        "Qwen/Qwen2.5-Omni-3B": "12 GB",
        "Salesforce/blip-vqa-base": "2 GB",
        "lmms-lab/LLaVA-Video-7B-Qwen2": "14 GB"
    }
    return size_estimates.get(model_name, "Unknown")

def download_specific_model(model_name, model_info):
    """
    ดาวน์โหลดโมเดลและส่วนประกอบที่จำเป็น (แก้ไข Memory Leak)
    """
    import gc  # เพิ่ม garbage collection
    
    save_path = os.path.join(BASE_SAVE_DIRECTORY, model_info["sub_dir"])
    
    logging.info(f"=" * 60)
    logging.info(f"Processing: {model_info.get('description', model_name)}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Estimated size: {estimate_download_size(model_name)}")
    logging.info(f"Target directory: {save_path}")
    logging.info(f"=" * 60)

    # ตรวจสอบว่าโมเดลมีอยู่แล้วหรือไม่
    config_file_path = os.path.join(save_path, 'config.json')
    if os.path.exists(config_file_path):
        logging.info("✓ Model already exists locally. Skipping download.")
        
        # ตรวจสอบไฟล์พิเศษสำหรับ Qwen-Omni
        if model_info.get("special_files"):
            download_special_files(model_name, save_path, model_info["special_files"])
        
        return True

    logging.info("Model not found locally. Starting download...")
    start_time = time.time()
    
    # สร้างตัวแปรเพื่อเก็บ references
    model = None
    processor = None
    
    try:
        os.makedirs(save_path, exist_ok=True)

        # ดาวน์โหลดด้วย snapshot_download สำหรับโมเดลที่ระบุ
        if model_info.get("download_method") == "snapshot":
            logging.info(f"Using snapshot download for '{model_name}'")
            snapshot_download(
                repo_id=model_name, 
                local_dir=save_path, 
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", "*.txt", "*.gitignore"]
            )
            
            # ดาวน์โหลดไฟล์พิเศษ
            if model_info.get("special_files"):
                download_special_files(model_name, save_path, model_info["special_files"])
        
        else:
            # ดาวน์โหลดแบบเดิมสำหรับโมเดลอื่นๆ
            model_loader_class = model_info["model_loader"]
            processor_loader_class = model_info.get("processor_loader", AutoProcessor)

            logging.info(f"Downloading processor/tokenizer...")
            processor = processor_loader_class.from_pretrained(
                model_name, 
                trust_remote_code=True,
                resume_download=True
            )
            processor.save_pretrained(save_path)
            logging.info("✓ Processor/tokenizer downloaded")
            
            # ล้าง processor จาก memory
            del processor
            processor = None
            gc.collect()  # บังคับ garbage collection

            logging.info(f"Downloading model weights... (This may take a while)")
            model = model_loader_class.from_pretrained(
                model_name, 
                torch_dtype="auto", 
                trust_remote_code=True,
                resume_download=True,
                device_map=None,  # ไม่โหลดไปที่ GPU
                low_cpu_mem_usage=True  # ใช้ memory น้อยลง
            )
            model.save_pretrained(save_path)
            logging.info("✓ Model weights downloaded")
            
            # ล้าง model จาก memory
            del model
            model = None
            gc.collect()  # บังคับ garbage collection
        
        # ล้าง GPU cache ถ้ามี
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("✓ GPU cache cleared")
        
        # คำนวณเวลาที่ใช้
        elapsed_time = time.time() - start_time
        logging.info(f"✓ Successfully downloaded '{model_name}' in {elapsed_time:.1f} seconds")
        
        # ล้าง memory เพิ่มเติม
        gc.collect()
        
        return True

    except Exception as e:
        logging.error(f"✗ Error downloading '{model_name}': {e}")
        logging.error("Troubleshooting tips:")
        logging.error("  1. Check internet connection")
        logging.error("  2. Verify disk space (need ~50GB total)")
        logging.error("  3. Update dependencies: pip install --upgrade transformers huggingface_hub")
        logging.error("  4. Try setting HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads")
        return False
    
    finally:
        # ล้าง memory ใน finally block เพื่อให้แน่ใจ
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        
        # ล้าง GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# เพิ่มฟังก์ชันสำหรับล้าง memory หลัง download เสร็จ
def cleanup_memory():
    """ล้าง memory หลังจาก download เสร็จ"""
    import gc
    
    logging.info("🧹 Cleaning up memory...")
    
    # บังคับ garbage collection หลายรอบ
    for i in range(3):
        collected = gc.collect()
        logging.info(f"  Garbage collection round {i+1}: freed {collected} objects")
    
    # ล้าง GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("  ✓ GPU cache cleared")
    
    # แสดง memory usage หลังล้าง
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        logging.info(f"  Current memory usage: {memory_mb:.1f} MB")
    except ImportError:
        pass
    
    logging.info("✓ Memory cleanup completed")

# แก้ไขฟังก์ชัน main()
def main():
    """
    ฟังก์ชันหลักที่จะวนลูปดาวน์โหลดโมเดลทั้งหมด
    """
    logging.info("🚀 Starting model download process...")
    
    # ตรวจสอบทรัพยากรระบบ
    check_system_resources()
    
    # สร้างโฟลเดอร์หลัก
    if not os.path.exists(BASE_SAVE_DIRECTORY):
        os.makedirs(BASE_SAVE_DIRECTORY)
        logging.info(f"Created base directory: {BASE_SAVE_DIRECTORY}")

    # เรียงลำดับโมเดลตาม priority
    sorted_models = sorted(
        MODELS_TO_DOWNLOAD.items(),
        key=lambda x: x[1].get("priority", 999)
    )
    
    successful_downloads = 0
    total_models = len(sorted_models)
    
    for model_name, model_info in sorted_models:
        if download_specific_model(model_name, model_info):
            successful_downloads += 1
        
        # ล้าง memory หลังแต่ละโมเดล
        cleanup_memory()
        
        # พักหายใจระหว่างดาวน์โหลด
        time.sleep(2)
        
    logging.info("=" * 60)
    logging.info(f"📊 Download Summary:")
    logging.info(f"  Successful: {successful_downloads}/{total_models}")
    logging.info(f"  Failed: {total_models - successful_downloads}/{total_models}")
    
    if successful_downloads == total_models:
        logging.info("🎉 All models downloaded successfully!")
    else:
        logging.warning("⚠️  Some models failed to download. Check logs above.")
    
    # ล้าง memory ครั้งสุดท้าย
    cleanup_memory()
    
    logging.info("🏁 Model download process finished.")

if __name__ == "__main__":
    main()