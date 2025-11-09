<div id="top">
		
<!-- HEADER STYLE: CLASSIC -->
<div align="center">
	<h1> Credit Risk Default Score Prediction Project <h1>
</div>
<!-- <img src="readmeai/assets/logos/purple.svg" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/> -->

<!-- # <code>❯ ## 🧩 </code> -->

<em></em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/JSON-000000.svg?style=default&logo=JSON&logoColor=white" alt="JSON">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=default&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=default&logo=tqdm&logoColor=black" alt="tqdm">
<img src="https://img.shields.io/badge/FastAPI-009688.svg?style=default&logo=FastAPI&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/MLflow-0194E2.svg?style=default&logo=MLflow&logoColor=white" alt="MLflow">
<br>
<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=default&logo=Docker&logoColor=white" alt="Docker">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Plotly-3F4F75.svg?style=default&logo=Plotly&logoColor=white" alt="Plotly">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=default&logo=YAML&logoColor=white" alt="YAML">

</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview
Credit Default Prediction — A machine learning project that predicts the likelihood of customer loan default to help financial institutions minimize credit risk. Built with Python, Scikit-learn, and Pandas, the model enhances data-driven lending and supports proactive risk mitigation.

Key Features:

Built predictive models using supervised learning algorithms (
		- Linear Regression
        - KNN
        - Decision Tree
        - Random Forest
        - Stacking Regressor (using the above models as base estimators)).

Performed data cleaning, feature engineering, and model evaluation to ensure accuracy and reliability

Generated actionable insights to support data-driven lending decisions and default mitigation strategies

Tools & Technologies: Python, Scikit-learn, Pandas, NumPy, Matplotlib, Jupyter Notebook, Mlflow

Outcome:
Improved prediction accuracy of default risk, enabling proactive financial decision-making and efficient credit portfolio management.

---

## Features

<code>❯ 🗒️📌 Key Features Used

| Feature Name             | Description                                      |
|--------------------------|--------------------------------------------------|
| income                   | Annual income of the applicant                   |
| loan_amount              | Total amount of the requested loan               |
| loan_term                | Duration of the loan in months                   |
| interest_rate            | Interest rate applied to the loan                |
| debt_to_income_ratio     | Ratio of total debt to annual income             |
| employment_years         | Years employed at current job                    |
| savings_balance          | Current savings balance                          |
| age                      | Age of the applicant                             | 
</code>

---

## Project Structure

```sh
└── /
    ├── .github
    │   └── workflows
    ├── api
    │   └── main.py
    ├── app.py
    ├── artifacts
    │   ├── data_ingestion
    │   ├── data_transformation
    │   ├── model_evaluation
    │   └── model_trainer
    ├── config
    │   └── config.yaml
    ├── Dockerfile
    ├── LICENSE
    ├── logs
    │   └── running_logs.log
    ├── main.py
    ├── notebook
    │   ├── data
    │   ├── EDA_DEFAULTPRED.ipynb
    │   ├── MODEL_TRAINER.ipynb
    │   └── trials.ipynb
    ├── params.yaml
    ├── README.md
    ├── requirements-streamlit.txt
    ├── requirements.txt
    ├── setup.py
    ├── src
    │   ├── defaultMlProj
    │   └── defaultMlProj.egg-info
    ├── template.py
    └── templates
        └── index.html
```

### Project Index

<details open>
	<summary><b><code>/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/app.py'>app.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ Setup our frontend with streamlit</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Dockerfile'>Dockerfile</a></b></td>
					<td style='padding: 8px;'>Code>❯ Build image for model deployment</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/LICENSE'>LICENSE</a></b></td>
					<td style='padding: 8px;'>Code>❯ MIT License</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/main.py'>main.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ Executes the data stages</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/params.yaml'>params.yaml</a></b></td>
					<td style='padding: 8px;'>Code>❯ Holds the model parameters</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/requirements-streamlit.txt'>requirements-streamlit.txt</a></b></td>
					<td style='padding: 8px;'>Code>❯ Necessary libraries for running steamlit frontend</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/requirements.txt'>requirements.txt</a></b></td>
					<td style='padding: 8px;'>Code>❯ Libraries for running the project</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/setup.py'>setup.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ Project setup</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/template.py'>template.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ Project template file</code></td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- api Submodule -->
	<details>
		<summary><b>api</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ api</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/api/main.py'>main.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ FASTAPI</code></td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- config Submodule -->
	<details>
		<summary><b>config</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ config</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/config/config.yaml'>config.yaml</a></b></td>
					<td style='padding: 8px;'>Code>❯ Configuration file</code></td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- notebook Submodule -->
	<details>
		<summary><b>notebook</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ notebook</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/notebook/EDA_DEFAULTPRED.ipynb'>EDA_DEFAULTPRED.ipynb</a></b></td>
					<td style='padding: 8px;'>Code>❯ Notebook</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/notebook/MODEL_TRAINER.ipynb'>MODEL_TRAINER.ipynb</a></b></td>
					<td style='padding: 8px;'>Code>❯ Model trainer notebook</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/notebook/trials.ipynb'>trials.ipynb</a></b></td>
					<td style='padding: 8px;'>Code>❯ Trials notebook</code></td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- templates Submodule -->
	<details>
		<summary><b>templates</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ templates</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/templates/index.html'>index.html</a></b></td>
					<td style='padding: 8px;'>Code>❯ HTML file</code></td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- artifacts Submodule -->
	<details>
		<summary><b>artifacts</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ artifacts</b></code>
			<!-- model_evaluation Submodule -->
			<details>
				<summary><b>model_evaluation</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ artifacts.model_evaluation</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/artifacts/model_evaluation/metrics.json'>metrics.json</a></b></td>
							<td style='padding: 8px;'>Code>❯ Store the model metrics</code></td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- model_trainer Submodule -->
			<details>
				<summary><b>model_trainer</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ artifacts.model_trainer</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/artifacts/model_trainer/model.joblib'>model.joblib</a></b></td>
							<td style='padding: 8px;'>Code>❯ Store model artifacts</code></td>
						</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<!-- src Submodule -->
	<details>
		<summary><b>src</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ src</b></code>
			<!-- defaultMlProj.egg-info Submodule -->
			<details>
				<summary><b>defaultMlProj.egg-info</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ src.defaultMlProj.egg-info</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/src/defaultMlProj.egg-info/dependency_links.txt'>dependency_links.txt</a></b></td>
							<td style='padding: 8px;'>Code>❯ Project pypi package links</code></td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/src/defaultMlProj.egg-info/PKG-INFO'>PKG-INFO</a></b></td>
							<td style='padding: 8px;'>Code>❯ Project pypi package</code></td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/src/defaultMlProj.egg-info/SOURCES.txt'>SOURCES.txt</a></b></td>
							<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/src/defaultMlProj.egg-info/top_level.txt'>top_level.txt</a></b></td>
							<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- defaultMlProj Submodule -->
			<details>
				<summary><b>defaultMlProj</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ src.defaultMlProj</b></code>
					<!-- components Submodule -->
					<details>
						<summary><b>components</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.defaultMlProj.components</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/components/data_ingestion.py'>data_ingestion.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Data ingestion file</code></td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/components/data_transformation.py'>data_transformation.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Data transformation file</code></td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/components/model_evaluation.py'>model_evaluation.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Model evaluation file</code></td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/components/model_trainer.py'>model_trainer.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Model trainer file</code></td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- config Submodule -->
					<details>
						<summary><b>config</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.defaultMlProj.config</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/config/configuration.py'>configuration.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Config file</code></td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- constants Submodule -->
					<details>
						<summary><b>constants</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.defaultMlProj.constants</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/constants/constant.py'>constant.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Project constants file</code></td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- entity Submodule -->
					<details>
						<summary><b>entity</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.defaultMlProj.entity</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/entity/config_entity.py'>config_entity.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Project Config file</code></td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- pipeline Submodule -->
					<details>
						<summary><b>pipeline</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.defaultMlProj.pipeline</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/pipeline/prediction.py'>prediction.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Model Prediction file</code></td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/pipeline/stage_data_ingestion.py'>stage_data_ingestion.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Data ingestion stage file</code></td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/pipeline/stage_data_transformation.py'>stage_data_transformation.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Data transformation stage file</code></td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/pipeline/stage_model_evaluation.py'>stage_model_evaluation.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Model evaluation stage file</code></td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/pipeline/stage_model_trainer.py'>stage_model_trainer.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Model tariner stage file</code></td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- utils Submodule -->
					<details>
						<summary><b>utils</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.defaultMlProj.utils</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/src/defaultMlProj/utils/common.py'>common.py</a></b></td>
									<td style='padding: 8px;'>Code>❯ Common functions project file</code></td>
								</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Pip
- **Container Runtime:** Docker

### Installation

Build  from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone ../
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd 
    ```

3. **Install the dependencies:**


	**Using [docker](https://www.docker.com/):**

	```sh
	❯ docker build -t / .
	```

	**Using [pip](https://pypi.org/project/pip/):**

	```sh
	❯ pip install -r requirements.txt
	```

### Usage

Run the project with:

**Using [docker](https://www.docker.com/):**
```sh
docker run -it {image_name}
```
**Using [pip](https://pypi.org/project/pip/):**
```sh
python {entrypoint}
```

### Testing

 uses the {__test_framework__} test framework. Run the test suite with:

**Using [pip](https://pypi.org/project/pip/):**
```sh
pytest
```

---

## Roadmap

- [ ] **`Task 1`**: Perform Data ingestion (Ingesting data either from local-csv/database/API).
- [ ] **`Task 2`**: Perform Data transformation (train/test split, filling missing values, encoding).
- [ ] **`Task 3`**: Train the model (Models training).
- [ ] **`Task 4`**: Perform Model evaluation (Evaluate model;
		metrics = {
                "r2_score": r2,
                "rmse": rmse,
                "mae": mae
            }
			Also setup Mlflow for model & parameter tracking
			).

---
### Evaluation Metrics

Each model was evaluated using:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared (R²) Score**

The **best performing model** selected based on these metrics.

- ✅ Stacking Regressor

---

## Contributing

- **💬 [Join the Discussions](https://LOCAL///discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://LOCAL///issues)**: Submit bugs found or log feature requests for the `` project.
- **💡 [Submit Pull Requests](https://LOCAL///blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your LOCAL account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone .
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to LOCAL**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://LOCAL{///}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=/">
   </a>
</p>
</details>

---

## License

 is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## Acknowledgments

- Credit `contributors`, `inspiration`, `references`, etc.

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---
