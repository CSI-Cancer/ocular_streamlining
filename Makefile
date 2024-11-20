#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON=python3
PIP=pip3
VENV_DIR=streamlining
PROJECT_DIR = $(CURDIR)
#################################################################################
# COMMANDS                                                                      #
#################################################################################
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/$(PIP) install --upgrade pip

install: venv
	$(VENV_DIR)/bin/$(PIP) install -r requirements.txt

prepare_data:
	PYTHONPATH=$(PROJECT_DIR)	$(VENV_DIR)/bin/$(PYTHON) streamlining_training/dataset.py

feature_select:
	PYTHONPATH=$(PROJECT_DIR)	$(VENV_DIR)/bin/$(PYTHON) streamlining_training/features.py

train:
	PYTHONPATH=$(PROJECT_DIR)	$(VENV_DIR)/bin/$(PYTHON) streamlining_training/train.py

evaluate:
	PYTHONPATH=$(PROJECT_DIR)	$(VENV_DIR)/bin/$(PYTHON) streamlining_training/eval.py