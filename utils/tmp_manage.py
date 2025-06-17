# tmp_manage.py
"""
Temporary File Management Module
จัดการไฟล์ชั่วคราวสำหรับ Qwen-Omni API
"""

import os
import time
import logging
from typing import Optional, List, Dict

# กำหนด temp directory ของโปรเจค
PROJECT_TEMP_DIR = "./tmp"

def ensure_temp_directory() -> None:
    """สร้างโฟลเดอร์ temp ถ้ายังไม่มี"""
    if not os.path.exists(PROJECT_TEMP_DIR):
        os.makedirs(PROJECT_TEMP_DIR)
        logging.info(f"Created temp directory: {PROJECT_TEMP_DIR}")

def get_temp_file_path(prefix: str, suffix: str) -> str:
    """
    สร้าง path สำหรับไฟล์ชั่วคราวในโฟลเดอร์โปรเจค
    
    Args:
        prefix: คำนำหน้าชื่อไฟล์ เช่น 'qwen_audio'
        suffix: นามสกุลไฟล์ เช่น '.wav'
    
    Returns:
        path ของไฟล์ temp
    """
    ensure_temp_directory()
    filename = f"{prefix}_{os.getpid()}_{int(time.time())}{suffix}"
    return os.path.join(PROJECT_TEMP_DIR, filename)

def safe_remove_file(filepath: str) -> bool:
    """
    ลบไฟล์อย่างปลอดภัย
    
    Args:
        filepath: path ของไฟล์ที่ต้องการลบ
    
    Returns:
        True ถ้าลบสำเร็จ, False ถ้าไม่สำเร็จ
    """
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logging.info(f"Cleaned up temp file: {filepath}")
            return True
    except Exception as e:
        logging.warning(f"Could not remove temp file {filepath}: {e}")
    return False

def cleanup_old_temp_files(max_age_hours: float = 1.0) -> int:
    """
    ทำความสะอาดไฟล์ temp เก่าๆ
    
    Args:
        max_age_hours: อายุสูงสุดของไฟล์ (ชั่วโมง) ที่จะเก็บไว้
    
    Returns:
        จำนวนไฟล์ที่ถูกลบ
    """
    try:
        if not os.path.exists(PROJECT_TEMP_DIR):
            return 0
            
        current_time = time.time()
        cleaned_count = 0
        
        for filename in os.listdir(PROJECT_TEMP_DIR):
            filepath = os.path.join(PROJECT_TEMP_DIR, filename)
            if os.path.isfile(filepath):
                # ตรวจสอบอายุไฟล์
                file_age = current_time - os.path.getctime(filepath)
                if file_age > (max_age_hours * 3600):  # แปลงชั่วโมงเป็นวินาที
                    if safe_remove_file(filepath):
                        cleaned_count += 1
        
        if cleaned_count > 0:
            logging.info(f"Cleaned up {cleaned_count} old temp files")
            
        return cleaned_count
            
    except Exception as e:
        logging.warning(f"Error during temp file cleanup: {e}")
        return 0

def get_temp_status() -> Dict:
    """
    ดูสถานะไฟล์ temp ทั้งหมด
    
    Returns:
        dictionary ข้อมูลสถานะ temp files
    """
    try:
        if not os.path.exists(PROJECT_TEMP_DIR):
            return {
                "temp_directory_exists": False,
                "temp_directory": PROJECT_TEMP_DIR,
                "total_files": 0,
                "total_size_mb": 0,
                "files": []
            }
        
        files_info = []
        total_size = 0
        
        for filename in os.listdir(PROJECT_TEMP_DIR):
            filepath = os.path.join(PROJECT_TEMP_DIR, filename)
            if os.path.isfile(filepath):
                file_size = os.path.getsize(filepath)
                file_age = time.time() - os.path.getctime(filepath)
                
                files_info.append({
                    "filename": filename,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024*1024), 2),
                    "age_minutes": round(file_age / 60, 1),
                    "age_hours": round(file_age / 3600, 2),
                    "path": filepath
                })
                total_size += file_size
        
        return {
            "temp_directory_exists": True,
            "temp_directory": os.path.abspath(PROJECT_TEMP_DIR),
            "total_files": len(files_info),
            "total_size_mb": round(total_size / (1024*1024), 2),
            "files": sorted(files_info, key=lambda x: x["age_minutes"])  # เรียงตามอายุ
        }
        
    except Exception as e:
        logging.error(f"Error getting temp status: {e}")
        return {
            "temp_directory_exists": False,
            "error": str(e),
            "files": []
        }

def cleanup_all_temp_files() -> Dict:
    """
    ลบไฟล์ temp ทั้งหมด
    
    Returns:
        dictionary ผลลัพธ์การลบ
    """
    try:
        if not os.path.exists(PROJECT_TEMP_DIR):
            return {
                "message": "Temp directory does not exist",
                "cleaned_files": 0,
                "total_size_mb": 0
            }
        
        cleaned_count = 0
        total_size = 0
        
        for filename in os.listdir(PROJECT_TEMP_DIR):
            filepath = os.path.join(PROJECT_TEMP_DIR, filename)
            if os.path.isfile(filepath):
                file_size = os.path.getsize(filepath)
                if safe_remove_file(filepath):
                    cleaned_count += 1
                    total_size += file_size
        
        # ลบโฟลเดอร์ถ้าว่าง
        try:
            if not os.listdir(PROJECT_TEMP_DIR):
                os.rmdir(PROJECT_TEMP_DIR)
                logging.info("Removed empty temp directory")
        except:
            pass
        
        return {
            "message": "Temp files cleaned successfully",
            "cleaned_files": cleaned_count,
            "total_size_mb": round(total_size / (1024*1024), 2)
        }
        
    except Exception as e:
        logging.error(f"Error cleaning all temp files: {e}")
        return {
            "error": f"Cleanup failed: {str(e)}",
            "cleaned_files": 0
        }

def initialize_temp_system(max_age_hours: float = 1.0) -> bool:
    """
    เริ่มต้นระบบ temp files
    
    Args:
        max_age_hours: อายุสูงสุดของไฟล์เก่าที่จะลบ
    
    Returns:
        True ถ้าเริ่มต้นสำเร็จ
    """
    try:
        ensure_temp_directory()
        cleaned_count = cleanup_old_temp_files(max_age_hours)
        
        status = get_temp_status()
        logging.info(f"✅ Temp system initialized:")
        logging.info(f"   Directory: {status['temp_directory']}")
        logging.info(f"   Current files: {status['total_files']}")
        logging.info(f"   Total size: {status['total_size_mb']} MB")
        if cleaned_count > 0:
            logging.info(f"   Cleaned old files: {cleaned_count}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize temp system: {e}")
        return False

class TempFileManager:
    """Context manager สำหรับจัดการไฟล์ temp"""
    
    def __init__(self, prefix: str, suffix: str):
        self.prefix = prefix
        self.suffix = suffix
        self.filepath = None
        self.created_files = []
    
    def __enter__(self):
        self.filepath = get_temp_file_path(self.prefix, self.suffix)
        return self.filepath
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.filepath:
            safe_remove_file(self.filepath)
    
    def create_additional_file(self, prefix: str, suffix: str) -> str:
        """สร้างไฟล์ temp เพิ่มเติมใน context นี้"""
        filepath = get_temp_file_path(prefix, suffix)
        self.created_files.append(filepath)
        return filepath
    
    def cleanup_all(self):
        """ลบไฟล์ทั้งหมดที่สร้างใน context นี้"""
        if self.filepath:
            safe_remove_file(self.filepath)
        
        for filepath in self.created_files:
            safe_remove_file(filepath)
        
        self.created_files.clear()


# Utility functions
def get_temp_dir_path() -> str:
    """ดึง path ของ temp directory"""
    return os.path.abspath(PROJECT_TEMP_DIR)

def is_temp_file(filepath: str) -> bool:
    """ตรวจสอบว่าไฟล์อยู่ใน temp directory หรือไม่"""
    try:
        temp_dir = os.path.abspath(PROJECT_TEMP_DIR)
        file_dir = os.path.dirname(os.path.abspath(filepath))
        return file_dir == temp_dir
    except:
        return False

def get_file_age_minutes(filepath: str) -> float:
    """ดึงอายุของไฟล์เป็นนาที"""
    try:
        if os.path.exists(filepath):
            return (time.time() - os.path.getctime(filepath)) / 60
    except:
        pass
    return 0.0


# สำหรับการทดสอบ
if __name__ == "__main__":
    # ทดสอบ functions
    print("Testing tmp_manage module...")
    
    # ทดสอบสร้างไฟล์
    test_file = get_temp_file_path("test", ".txt")
    with open(test_file, "w") as f:
        f.write("Test content")
    print(f"Created test file: {test_file}")
    
    # ทดสอบ status
    status = get_temp_status()
    print(f"Temp status: {status}")
    
    # ทดสอบ cleanup
    cleanup_all_temp_files()
    print("Cleanup completed")
    
    # ทดสอบ context manager
    with TempFileManager("context_test", ".dat") as temp_file:
        with open(temp_file, "w") as f:
            f.write("Context manager test")
        print(f"Using context manager: {temp_file}")
    print("Context manager cleanup completed")