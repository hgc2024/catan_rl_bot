
from tests.test_phase1 import TestPhase1
import unittest
import traceback

t = TestPhase1()
t.setUp()
try:
    t.test_step_structure()
    print("Test passed!")
except Exception:
    traceback.print_exc()
