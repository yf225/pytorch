from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.11.0.dev20260109+cu130'
debug = False
cuda: Optional[str] = '13.0'
git_version = '2af01790ba14ad456bb624e5bffe313cf611cce4'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
