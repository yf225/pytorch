from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.11.0.dev20260107+cu130'
debug = False
cuda: Optional[str] = '13.0'
git_version = '4f0b239c7757d538e605e6431aa5bd05d78a54f9'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
