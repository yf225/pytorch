from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.11.0.dev20251220+cu130'
debug = False
cuda: Optional[str] = '13.0'
git_version = '43c9b64d4586b39169b7ae62997cfa73c45a6bc4'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
