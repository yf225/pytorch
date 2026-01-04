from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.11.0.dev20260104+cu128'
debug = False
cuda: Optional[str] = '12.8'
git_version = 'c6bcb25d8bfe09da372676d68d3502d9e57fef7b'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
