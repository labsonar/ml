"""
Setup Script
============

This module provides the setup configuration for the `project_name` package.

Setup scripts should contain:
- Configuration of package metadata such as name, version, description, authors, and license.
- Specification of package requirements and additional package data if needed.

This example demonstrates the structure and usage of a setup script but can be used as default
setup file.
"""

import os
import setuptools

# Using the grandparent directory (git repository) as package name
grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_name = os.path.basename(grandparent_dir).lower()

# Setup configuration for the package
setuptools.setup(
    name=package_name,
    version='0.0.1',
    packages=setuptools.find_packages(),
    description='A simple Python project setup example.',
    author='Author Name',
    license='CC BY-NC-SA 4.0',
    install_requires=[
        # List your project dependencies here
    ],
    # Uncomment and specify package data if needed
    # package_data={package_name: ['info/*.csv']},
)
