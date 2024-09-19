"""
Example Unit Test
=================

This module contains unit tests for the `example` module in the `project_name` package.

Unit test files should contain:
- Imports of necessary modules and the module under test.
- One or more test class derived from `unittest.TestCase`.
- Test methods within the test class to verify the functionality of the module under test.

This example demonstrates the structure and usage of a unit test file.
"""

import unittest
import project_name.example as lib

class ExampleUnitTest(unittest.TestCase):
    """
    Unit test class for the example module.

    This class contains tests for the functions and classes in the example module.
    """

    def test_sample_function(self):
        """
        Test the sample_function to ensure it returns the expected value (True).
        """
        self.assertTrue(lib.sample_function())

if __name__ == '__main__':
    unittest.main()
