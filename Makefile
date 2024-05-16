ROOT_DIR = $(shell pwd)
SCRIPTS_DIR = $(ROOT_DIR)/scripts
DATASETS ?=  'default'

setup:
	pip install -r requirements.txt

histogram-thresholding:
	python -m scripts.histogram-thresholding.src.histogram-thresholding --datasets $(DATASETS)

segment-plastic-base:
	python -m scripts.segment-plastic-base.src.segment-plastic-base --datasets $(DATASETS)

segment-plastic-cell-edges:
	python -m scripts.segment-plastic-cell-edges.src.segment-plastic-cell-edges --datasets $(DATASETS)

chain-task-directories:
	-rm -rf $(SCRIPTS_DIR)/segment-plastic-base/input
	ln -s $(SCRIPTS_DIR)/histogram-thresholding/output $(SCRIPTS_DIR)/segment-plastic-base/input
	-rm -rf $(SCRIPTS_DIR)/segment-plastic-cell-edges/input
	ln -s $(SCRIPTS_DIR)/segment-plastic-base/output $(SCRIPTS_DIR)/segment-plastic-cell-edges/input

unchain-task-directories:
	-unlink $(SCRIPTS_DIR)/segment-plastic-base/input
	mkdir $(SCRIPTS_DIR)/segment-plastic-base/input
	-unlink $(SCRIPTS_DIR)/segment-plastic-cell-edges/input
	mkdir $(SCRIPTS_DIR)/segment-plastic-cell-edges/input

run: histogram-thresholding segment-plastic-base segment-plastic-cell-edges

clean: unchain-task-directories
	-rm -rf $(SCRIPTS_DIR)/histogram-thresholding/input/
	-rm -rf $(SCRIPTS_DIR)/segment-plastic-base/input/
	-rm -rf $(SCRIPTS_DIR)/segment-plastic-cell-edges/input/
	-rm -rf $(SCRIPTS_DIR)/histogram-thresholding/output/
	-rm -rf $(SCRIPTS_DIR)/segment-plastic-base/output/
	-rm -rf $(SCRIPTS_DIR)/segment-plastic-cell-edges/output/
	mkdir $(SCRIPTS_DIR)/histogram-thresholding/input/
	mkdir $(SCRIPTS_DIR)/segment-plastic-base/input/
	mkdir $(SCRIPTS_DIR)/segment-plastic-cell-edges/input/
	mkdir $(SCRIPTS_DIR)/histogram-thresholding/output/
	mkdir $(SCRIPTS_DIR)/segment-plastic-base/output/
	mkdir $(SCRIPTS_DIR)/segment-plastic-cell-edges/output/
