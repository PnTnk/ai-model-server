# src/utils/__init__.py
"""
Utils package for Qwen-Omni API

Provides utility functions for:
- Temporary file management (tmp_manage)
- Model downloading (download_model)
"""

# Import all functions from tmp_manage
from .tmp_manage import (
    get_temp_file_path,
    safe_remove_file,
    cleanup_old_temp_files,
    get_temp_status,
    cleanup_all_temp_files,
    initialize_temp_system,
    TempFileManager,
    get_temp_dir_path,
    is_temp_file,
    get_file_age_minutes
)

# Import functions from download_model if needed
try:
    from .download_model import (
        download_specific_model,
        MODELS_TO_DOWNLOAD
    )
except ImportError:
    # download_model might not have these functions
    pass

__version__ = "1.0.0"

__all__ = [
    # tmp_manage functions
    'get_temp_file_path',
    'safe_remove_file',
    'cleanup_old_temp_files', 
    'get_temp_status',
    'cleanup_all_temp_files',
    'initialize_temp_system',
    'TempFileManager',
    'get_temp_dir_path',
    'is_temp_file',
    'get_file_age_minutes',
    
    # download_model functions (if available)
    'download_specific_model',
    'MODELS_TO_DOWNLOAD'
]

# Package info
def get_package_info():
    """Get package information"""
    return {
        "name": "qwen-omni-utils",
        "version": __version__,
        "description": "Utility functions for Qwen-Omni API",
        "modules": ["tmp_manage", "download_model"]
    }