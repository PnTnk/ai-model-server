import os
import sys
import logging
import argparse
import subprocess
import psutil
import time
import gc
from pathlib import Path


def setup_logging():
    """ตั้งค่า logging สำหรับระบบ"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('./logs/main.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)


def ensure_directory_exists(directory):
    """
    ตรวจสอบว่าไดเรกทอรีที่ระบุมีอยู่หรือไม่ ถ้าไม่มีก็จะสร้างขึ้นมา
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"--- Created directory: {directory} ---")
    else:
        print(f"--- Directory already exists: {directory} ---")


def monitor_memory():
    """แสดง memory usage ปัจจุบัน"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        # ตรวจสอบ system memory
        system_memory = psutil.virtual_memory()
        available_gb = system_memory.available / (1024**3)
        used_percent = system_memory.percent
        
        return {
            'process_mb': memory_mb,
            'system_available_gb': available_gb,
            'system_used_percent': used_percent
        }
    except:
        return None


def kill_process_by_name(process_name):
    """ฆ่า process ตามชื่อ"""
    logger = logging.getLogger(__name__)
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if process_name in proc.info['name'] or any(process_name in cmd for cmd in (proc.info['cmdline'] or [])):
                proc.kill()
                logger.info(f"Killed process {proc.info['name']} (PID: {proc.info['pid']})")
                return True
        return False
    except Exception as e:
        logger.warning(f"Error killing process {process_name}: {e}")
        return False


def force_cleanup_memory():
    """บังคับล้าง memory"""
    logger = logging.getLogger(__name__)
    
    logger.info("🧹 Force cleaning memory...")
    
    # ล้าง Python garbage collection
    for i in range(5):
        collected = gc.collect()
        logger.info(f"  GC round {i+1}: freed {collected} objects")
    
    # ล้าง GPU cache ถ้ามี
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("  ✓ GPU cache cleared")
    except ImportError:
        pass
    
    # รอให้ระบบจัดการ memory
    time.sleep(3)
    
    memory_info = monitor_memory()
    if memory_info:
        logger.info(f"  Memory after cleanup: {memory_info['process_mb']:.1f} MB")


def check_system_requirements():
    """ตรวจสอบความต้องการของระบบ"""
    logger = logging.getLogger(__name__)
    
    # แสดง memory ปัจจุบัน
    memory_info = monitor_memory()
    if memory_info:
        logger.info(f"System memory - Used: {memory_info['system_used_percent']:.1f}%, Available: {memory_info['system_available_gb']:.1f} GB")
    
    try:
        import torch
        import flask
        import transformers
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Flask version: {flask.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # ตรวจสอบ CUDA
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            logger.warning("CUDA not available - models will run on CPU")
            
        return True
        
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False


def download_models_if_needed():
    """ดาวน์โหลดโมเดลถ้ายังไม่มี - ปรับปรุงการจัดการ memory"""
    logger = logging.getLogger(__name__)
    
    # ตรวจสอบว่ามีโมเดลอยู่แล้วหรือไม่
    models_dir = Path("./models")
    required_models = [
        "deepseek-8b",
        "qwen-vl-3b", 
        "qwen-audio-7b",
        "qwen-omni-3b",
        "blip-vqa-base"
    ]
    
    missing_models = []
    for model in required_models:
        model_path = models_dir / model / "config.json"
        if not model_path.exists():
            missing_models.append(model)
    
    if missing_models:
        logger.info(f"Missing models detected: {missing_models}")
        logger.info("Starting model download process...")
        
        # แสดง memory ก่อน download
        memory_before = monitor_memory()
        if memory_before:
            logger.info(f"Memory before download: {memory_before['process_mb']:.1f} MB")
        
        download_process = None
        try:
            # ตั้งค่า environment variables เพื่อลด memory usage
            env = os.environ.copy()
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            env['TRANSFORMERS_OFFLINE'] = '0'
            env['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            env['CUDA_VISIBLE_DEVICES'] = ''  # บังคับใช้ CPU เท่านั้น
            
            # รัน download_model.py ใน process แยก
            download_process = subprocess.Popen([
                sys.executable, "./utils/download_model.py"
            ], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
            )
            
            # Monitor process แบบ real-time
            start_time = time.time()
            timeout = 3600  # 1 hour timeout
            
            while download_process.poll() is None:
                # ตรวจสอบ timeout
                if time.time() - start_time > timeout:
                    logger.error("Download process timed out")
                    download_process.kill()
                    return False
                
                # แสดง memory usage ทุก 30 วินาที
                if int(time.time() - start_time) % 30 == 0:
                    memory_info = monitor_memory()
                    if memory_info:
                        logger.info(f"Download progress - Memory: {memory_info['process_mb']:.1f} MB, System: {memory_info['system_used_percent']:.1f}%")
                
                time.sleep(1)
            
            # รับผลลัพธ์
            stdout, stderr = download_process.communicate()
            
            if download_process.returncode == 0:
                logger.info("✓ Model download completed successfully")
                
                # แสดงผลลัพธ์สำคัญ
                if stdout:
                    for line in stdout.split('\n'):
                        if 'Successfully downloaded' in line or 'Download Summary' in line:
                            logger.info(f"  {line}")
                
                return True
            else:
                logger.error(f"✗ Model download failed (exit code: {download_process.returncode})")
                if stderr:
                    logger.error(f"Error output: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error during model download: {e}")
            if download_process:
                download_process.kill()
            return False
        
        finally:
            # ล้าง memory หลังจาก download เสร็จ
            logger.info("Cleaning up after model download...")
            
            # รอให้ process จบสมบูรณ์
            if download_process:
                try:
                    download_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    download_process.kill()
            
            # ฆ่า process ที่อาจค้างอยู่
            kill_process_by_name("download_model.py")
            
            # บังคับล้าง memory
            force_cleanup_memory()
            
            # แสดง memory หลังล้าง
            memory_after = monitor_memory()
            if memory_after and memory_before:
                memory_diff = memory_after['process_mb'] - memory_before['process_mb']
                logger.info(f"Memory after download: {memory_after['process_mb']:.1f} MB (diff: {memory_diff:+.1f} MB)")
    else:
        logger.info("✓ All required models are already available")
        return True


def start_flask_server(host="0.0.0.0", port=5000, debug=False):
    """เริ่มต้น Flask server"""
    logger = logging.getLogger(__name__)
    
    # แสดง memory ก่อนเริ่ม server
    memory_info = monitor_memory()
    if memory_info:
        logger.info(f"Memory before starting server: {memory_info['process_mb']:.1f} MB")
    
    try:
        # นำเข้า app จาก app.py
        from app import app
        
        logger.info(f"Starting Flask server on {host}:{port}")
        logger.info("Available endpoints:")
        logger.info("  POST /generate/deepseek - Text generation")
        logger.info("  POST /generate/qwen_vl - Vision + text")
        logger.info("  POST /generate/qwen_audio - Audio + text")
        logger.info("  POST /generate/qwen_omni - Multimodal (text/audio/video/image)")
        logger.info("  POST /generate/blip_vqa - Visual Q&A")
        logger.info("  GET /admin/temp_status - Check temp files")
        logger.info("  POST /admin/cleanup_temp - Cleanup temp files")
        
        # เริ่ม server
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
            use_reloader=False  # ปิดใช้งาน auto-reload ใน production
        )
        
    except ImportError as e:
        logger.error(f"Cannot import Flask app: {e}")
        logger.error("Make sure app.py exists in src/ directory")
        return False
    except Exception as e:
        logger.error(f"Error starting Flask server: {e}")
        return False


def main():
    """ฟังก์ชันหลัก"""
    # ตั้งค่า argument parser
    parser = argparse.ArgumentParser(description="AI Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download check")
    parser.add_argument("--download-only", action="store_true", help="Only download models, don't start server")
    parser.add_argument("--force-cleanup", action="store_true", help="Force memory cleanup before starting")
    
    args = parser.parse_args()
    
    # บังคับล้าง memory ถ้าระบุ
    if args.force_cleanup:
        force_cleanup_memory()
    
    # สร้างไดเรกทอรีที่จำเป็น
    directories_to_check = [
        "./models", 
        "./utils",
        "./tmp",
        "./logs"
    ]
    
    print("=== Setting up directories ===")
    for directory in directories_to_check:
        ensure_directory_exists(directory)
    
    # ตั้งค่า logging
    logger = setup_logging()
    logger.info("=== AI Model Server Starting ===")
    
    # ตรวจสอบ system requirements
    logger.info("Checking system requirements...")
    if not check_system_requirements():
        logger.error("System requirements not met. Exiting.")
        sys.exit(1)
    
    # ดาวน์โหลดโมเดลถ้าจำเป็น
    if not args.skip_download:
        logger.info("Checking and downloading models if needed...")
        if not download_models_if_needed():
            logger.error("Model download failed. Exiting.")
            sys.exit(1)
    
    # ถ้าเป็น download-only mode ให้หยุดที่นี่
    if args.download_only:
        logger.info("Download-only mode completed. Exiting.")
        # ล้าง memory ก่อนออก
        force_cleanup_memory()
        return
    
    # เริ่ม Flask server
    logger.info("All checks passed. Starting Flask server...")
    try:
        start_flask_server(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        logger.info("Cleaning up before exit...")
        force_cleanup_memory()
        logger.info("=== AI Model Server Stopped ===")


if __name__ == "__main__":
    main()