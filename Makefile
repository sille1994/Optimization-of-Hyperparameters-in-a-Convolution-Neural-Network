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
	conda env create --prefix ./envs --file environment.yml
term:
	rm -rf ./envsr
doc:
<<<<<<< HEAD
	pdoc --html --html-dir ./doc/ --overwrite ./Hyperparameter_Optimization/Running_HO_on_CNN_doc.py
=======
	pdoc3 --force --html --output-dir ./docs $(MODULENAME)
>>>>>>> 2787f59b35238fcc5f86451e91899a30d4742c3d

lint:
	pylint $(MODULENAME)

test:
	pytest -v $(MODULENAME)

runcode:
	python ./Hyperparameter_Optimization/Running_HO_on_CNN.py

clean:
	rm .gitkeep -f
	rm .DS_Store -f

.PHONY: init doc lint test 

