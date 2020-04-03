clean:
	rm .gitkeep -f
	rm .DS_Store -f

init:
	conda env create --prefix ./envs --file environment.yml

term:
	rm -rf ./envs

doc:
	pdoc --html --html-dir ./doc --overwrite ./Hyperparameter_Optimization/Running_HO_on_CNN_doc.py
	cd Hyperparameter_Optimization
	rm __pycache__ -f

lint:
	pylint --disable=no-member Hyperparameter_Optimization

test:
	python ./Hyperparameter_Optimization/test_Running_HO_on_CNN.py

runcode:
	python ./Hyperparameter_Optimization/Running_HO_on_CNN.py
