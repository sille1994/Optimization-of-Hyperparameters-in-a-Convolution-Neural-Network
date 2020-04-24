MODULENAME = Hyperparameter_Optimization 

help:
	@echo ""
	@echo "Welcome to my project!!!"
	@echo "To get started create an environment using:"
	@echo "	make init"
	@echo "	conda activate ./envs"
	@echo ""
	@echo "To generate project documentation use:"
	@echo "	make doc"
	@echo ""
	@echo "To Lint the project use:"
	@echo "	make lint"
	@echo ""
	@echo "To run unit tests use:"
	@echo "	make test"
	@echo ""
	

init:
	conda env create --prefix ./envs --file ./Software/environment.yml
term:
	rm -rf ./envs
doc:
	pdoc --html --html-dir ./doc --overwrite ./Hyperparameter_Optimization/Running_HO_on_CNN_doc.py


lint:
	pylint $(MODULENAME)

test:
	python ./Hyperparameter_Optimization/test_Running_HO_on_CNN.py

runcode:
	python ./Hyperparameter_Optimization/Running_HO_on_CNN.py

clean:
	rm .gitkeep -f
	rm .DS_Store -f

.PHONY: init doc lint test 

