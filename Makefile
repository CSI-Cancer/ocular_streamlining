#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON=python3.11
PROJECT_DIR = $(CURDIR)
#################################################################################
# COMMANDS                                                                      #
#################################################################################
.PHONY: install prepare_data feature_select train evaluate clean_data clean_venv
install:
	poetry	install

prepare_data:
	poetry	run prepare_data

feature_select:
	poetry	run feature_select

train:
	poetry	run train

evaluate:
	poetry	run evaluate

clean_data:
	rm -rf data/processed/*
	rm -rf data/interim/*

clean_venv:
	@venv_path=$$(poetry	env	info	--path)	&&	rm	-rf	$$venv_path

clean_models:
	rm	-rf	models/*