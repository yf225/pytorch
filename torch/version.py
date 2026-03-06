from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.12.0.dev20260306+cu130'
debug = False
cuda: Optional[str] = '13.0'
git_version = 'a9e2dc45400a26f2e5573e48f153d0ba3ca6ecda'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
