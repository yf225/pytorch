from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.11.0.dev20260202+cu130'
debug = False
cuda: Optional[str] = '13.0'
git_version = '56f6cd27d7510fc5d6306d6ce9edd345d7125902'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
