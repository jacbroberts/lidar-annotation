"""
SLAM Dataset Downloaders
"""

from .base_downloader import BaseDownloader
from .tum_downloader import TUMDownloader
from .icl_downloader import ICLNUIMDownloader
from .euroc_downloader import EuRoCDownloader
from .newer_college_downloader import NewerCollegeDownloader

__all__ = [
    'BaseDownloader',
    'TUMDownloader',
    'ICLNUIMDownloader',
    'EuRoCDownloader',
    'NewerCollegeDownloader',
]
