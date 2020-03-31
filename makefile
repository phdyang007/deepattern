fakegen:
	mkdir -p ./data/fake
	python3 ./src/fakegen.py

prepare%:
	python3 src/prepare.py /research/byu2/hyyang/tmp/to_haoyu_200303/rad96/$*/catalog_on_corners/db/deckProp.dbreduced.msgpack ./data/$*/

train%:
	mkdir -p models/$*/sample 
	mkdir -p models/$*/test
	python3 src/main.py ./data/$* ./models/$* &

test%:
	rm -rf models/test/*
	python3 src/test.py ./data/$* ./models/$* 
	python3 src/merge.py ./models/$*/ 
	rm -rf models/$*/test/noise_data_*.msgpack

gantrain%:
	mkdir -p models/$*/gan/
	python3 src/gan.py train $*
gangenerate%:
	mkdir -p models/$*/gan/test/
	python3 src/gan.py test $*
gantest%:
	rm -rf models/test/*
	python3 src/testgan.py ./data/$* ./models/$* 
	python3 src/merge.py ./models/$*/gan/
	rm -rf models/$*/gan/test/noise_data_*.msgpack

merge%:
	python3 src/merge.py ./models/$*/ 


eval%:
	python3 src/eval.py $* |& tee ./models/$*/test/$@.txt

cleangan%:
	rm -rf models/$*/gan/*


clean%:
	rm -rf models/$*

clean_all:
	rm -rf models/*




	
