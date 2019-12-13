euv:
	python src/main.py 
test:
	rm -rf models/test/*
	python src/test.py
clean:
	rm -rf models/step*
	rm -rf models/checkpoint
	rm -rf models/sample/*
	rm -rf models/test/*