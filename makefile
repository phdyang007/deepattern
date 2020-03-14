euv:
	mkdir -p models/sample
	mkdir -p models/test
	python3 src/main.py 
test:
	rm -rf models/test/*
	python3 src/test.py

df:
	python3 ./src/fakegen.py

tc:
	python3 src/prepare.py ../tmp/to_haoyu_200303/rad96/tc1/catalog_on_corners/db/deckProp.dbreduced.msgpack ./data/tc1/
	python3 src/prepare.py ../tmp/to_haoyu_200303/rad96/tc2/catalog_on_corners/db/deckProp.dbreduced.msgpack ./data/tc2/
	python3 src/prepare.py ../tmp/to_haoyu_200303/rad96/tc3/catalog_on_corners/db/deckProp.dbreduced.msgpack ./data/tc3/
	python3 src/prepare.py ../tmp/to_haoyu_200303/rad96/tc4/catalog_on_corners/db/deckProp.dbreduced.msgpack ./data/tc4/
	python3 src/prepare.py ../tmp/to_haoyu_200303/rad96/tc5/catalog_on_corners/db/deckProp.dbreduced.msgpack ./data/tc5/
	python3 src/prepare.py ../tmp/to_haoyu_200303/rad96/tc6/catalog_on_corners/db/deckProp.dbreduced.msgpack ./data/tc6/
	
clean:
	rm -rf models/step*
	rm -rf models/checkpoint
	rm -rf models/sample/*
	rm -rf models/test/*



	