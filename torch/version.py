from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.11.0.dev20260120+cu130'
debug = False
cuda: Optional[str] = '13.0'
git_version = 'd26b4532c0187cc727106785138a5b458bdb8fa0'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
