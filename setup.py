import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "End-to-end-Default-Risk-Pred-mlProject"
AUTHOR_USER_NAME = "klan86at"
SRC_REPO = "mlProject"
AUTHOR_EMAIL = "klang86at@gmail.com"

# This setup script is used to package the project for distribution.
setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A brief python package for ML application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
    # The package directory is set to 'src' to avoid issues with imports.
)