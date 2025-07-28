# Libraries
import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s ]:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MLProjectSetup")

project_name = "defaultMlProj"


list_of_files = [
    # GitHub Actions (optional, keep .gitkeep to track the folder)
    ".github/workflows/.gitkeep",

    # Source code structure
    f"src/{project_name}/__init__.py",

    # Core components (trimmed)
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",

    # Utility functions
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",

    # Configuration
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    # Pipeline
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",

    # Entity (config/data classes)
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",

    # Constants
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/constants/constant.py",

    # Exceptions
    f"src/{project_name}/exception/exception.py",

    # API
    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/main.py",
    f"src/{project_name}/api/routes/prediction.py",

    # Config files (YAML)
    Path("config") / "config.yaml",
    Path("config") / "params.yaml",
    Path("config") / "schema.yaml",

    # Entry points
    "main.py",
    "app.py",

    # Api
    "api/main.py"

    # Packaging and deployment
    "requirements.txt",
    "setup.py",

    # Research
    "notebook/trials.ipynb",

    # Tests
    f"tests/__init__.py",
    f"tests/test_data_ingestion.py",

    # Web templates (for Flask/Dash)
    "templates/index.html",
]

for filepath in list_of_files:
    path = Path(filepath)
    directory = path.parent
    filename = path.name

    # Create directory if needed
    if directory != Path("."):
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory} for file: {filename}")

    # Create file
    if not path.exists():
        path.touch()
        logger.info(f"Created file: {path}")