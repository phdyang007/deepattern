fakegen:
	mkdir -p ./data/fake
	python3 ./src/fakegen.py

prepare%:
	python3 src/prepare.py /research/byu2/hyyang/tmp/to_haoyu_200303/rad96/$*/catalog_on_corners/db/deckProp.dbreduced.msgpack ./data/$*/

train%:
	mkdir -p models/$*/sample 
	mkdir -p models/$*/test
	python3 src/main.py ./data/$* ./models/$*

test%:
	rm -rf models/test/*
	python3 src/test.py ./data/$* ./models/$*
	python3 src/merge.py ./models/$*/
	rm -rf models/test/noise_data_*.msgpack

eval%:
	python3 src/eval.py $* |& tee ./models/$*/test/$@.txt

clean%:
	rm -rf models/$*

clean_all:
	rm -rf models/*




	
