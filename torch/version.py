from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.11.0.dev20251230+cu130'
debug = False
cuda: Optional[str] = '13.0'
git_version = '9e197ae2e31506c371e8bf13580b7f93aa1a0378'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
