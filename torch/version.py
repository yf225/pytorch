from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.11.0.dev20260114+cu130'
debug = False
cuda: Optional[str] = '13.0'
git_version = 'cc5b657fef205df9e6c0d7d63be16fad4e5841e4'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
