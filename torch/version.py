from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.12.0.dev20260410+cu130'
debug = False
cuda: Optional[str] = '13.0'
git_version = '9f4fa548fb25fcf1b4a9fdf55c5bbbbe7e53cc12'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
