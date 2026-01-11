from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.11.0.dev20260111+cu130'
debug = False
cuda: Optional[str] = '13.0'
git_version = '7baf3702124f35ac8edd0a0c195bff40a2e1a911'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
