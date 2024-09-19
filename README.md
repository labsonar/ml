## Introduction
This project provides a basic template for developing Python libraries that promote uniformity and centralization within the LabSonar/LPS environment.

## Authors
- **Developer**: [Fábio Oliveira](https://github.com/obs-fabio)
- **Advisor**: [Natanael Junior](https://github.com/natmourajr/natmourajr)

## Repository Structure

The following directory structure is recommended for library development:

- **src/**: Contains the core Python code that forms the pip package.
- **unit_test/**: Contains simple scripts for unit testing, enabling focused validation of individual features within the package.
- **apps/**: Contains more complex scripts that leverage one or more features from the package. These scripts can be designed as command-line applications for practical use.
- **notebooks/**: If desired, this directory can be added to include Jupyter Notebooks to showcase demonstrations, tutorials, and interactive code examples.


## Code Standard
To ensure code maintainability, it is recommended that all developed code uses typing annotations and check the code with pylint before committing.
Typing helps in defining the expected data types of variables and functions, which improves code readability and helps catch errors early.
Pylint is a tool that checks for errors in Python code, enforces a coding standard, and looks for code smells.

## Installation

### Development Mode
To install the library in development mode, enabling real-time modifications during the development process, navigate to the `src` directory and execute the following command in your terminal:
```bash
pip install -e . --user
```
This command installs the library in editable mode, allowing you to make changes to the code within the `src` directory and observe them reflected without needing to reinstall the package.

### Deployment Mode
To install the library in regular installation mode, suitable for production environments or sharing with others, navigate to the `src` directory and run the following command:
```bash
pip install . --user
```

This command installs the library as a standalone package, making it accessible for use within your project or by others who have installed it.

To generate the pip package as a .tar.gz file, navigate to the `src` directory and run the following command. The output file will be generated in the dist directory:
```bash
python setup.py sdist bdist_wheel
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) license. You are free to use, modify, and distribute the code for non-commercial purposes, with the condition that you provide attribution to the authors and distribute any derivative works under the same license. For more details, please refer to the license file (LICENSE.md) included in this repository.
