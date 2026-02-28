from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.12.0.dev20260224+cu130'
debug = False
cuda: Optional[str] = '13.0'
git_version = 'aed20b1240f5fee4b93227769c79414f66b62ced'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
