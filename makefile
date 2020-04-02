fakegen:
	mkdir -p ./data/fake
	python3 ./src/fakegen.py

train_all: traintc1 traintc2 traintc3 traintc4 traintc5 traintc6
test_all: testtc1 testtc2 testtc3 testtc4 testtc5 testtc6
gantrain_all: gantraintc1 gantraintc2 gantraintc3 gantraintc4 gantraintc5 gantraintc6
gangenerate_all: gangeneratetc1 gangeneratetc2 gangeneratetc3 gangeneratetc4 gangeneratetc5 gangeneratetc6
gantest_all: gantesttc1 gantesttc2 gantesttc3 gantesttc4 gantesttc5 gantesttc6
eval_all: evaltc1 evaltc2 evaltc3 evaltc4 evaltc5 evaltc6	

tcae_%: 
	make train$*
	make test$*

gan_%: 
	make gangen$*
	make gantrain$* 
	make gantest$*

	
prepare%:
	python3 src/prepare.py /research/byu2/hyyang/tmp/to_haoyu_200303/rad96/$*/catalog_on_corners/db/deckProp.dbreduced.msgpack ./data/$*/

train%:
	mkdir -p models/$*/sample 
	mkdir -p models/$*/test
	python3 src/main.py ./data/$* ./models/$* 

test%:
	rm -rf models/test/*
	python3 src/test.py ./data/$* ./models/$* 
	python3 src/merge.py ./models/$*/ tcae
	rm -rf models/$*/test/noise_data_*.msgpack

gangen%:
	mkdir -p models/$*/gan/train
	python3 src/gangen.py ./data/$* ./models/$*
	python3 src/merge.py ./models/$*/gan/train gan
	rm -rf models/$*/gan/train/noise_data_*.msgpack

gantrain%:
	mkdir -p models/$*/gan/
	python3 src/gan.py train $*
	mkdir -p models/$*/gan/test/
	python3 src/gan.py test $*
gantest%:
	rm -rf models/test/*
	python3 src/testgan.py ./data/$* ./models/$* 
	python3 src/merge.py ./models/$*/gan/ tcae
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




	
