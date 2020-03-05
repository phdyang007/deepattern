2d:
	mkdir -p models/sample
	mkdir -p models/test
	python3 src/main.py 
test:
	rm -rf models/test/*
	python3 src/test.py

clean:
	rm -rf models/step*
	rm -rf models/checkpoint
	rm -rf models/sample/*
	rm -rf models/test/*



	
