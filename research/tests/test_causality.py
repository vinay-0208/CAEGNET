# -*- coding: utf-8 -*-
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(cur_dir, '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tests.test_causality import TestCAEGCausalityAndLeakage

if __name__ == '__main__':
    import unittest
    unittest.main()
