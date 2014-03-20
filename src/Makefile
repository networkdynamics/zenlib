
build:
	python setup.py build

install:
	python setup.py install

test:
	cd ..; python -m zen.test

benchmark:
	cd ..; python -m zen.benchmark

build_inplace:
	python setup.py build_ext --inplace

# Set argument test to the name to run a specific test
#  e.g. make test_inplace test="zen.tests.control.ProfileTestCase"
test_inplace:
	export PYTHONPATH=.:${PYTHONPATH}; python -m zen.test $(test)

benchmark_inplace:
	export PYTHONPATH=.:${PYTHONPATH}; python -m zen.benchmark

clean:
	rm -rf build; \
	rm -f zen/*.c zen/*.so zen/*.pyc; \
	rm -f zen/algorithms/*.c zen/algorithms/*.so zen/algorithms/*.pyc; \
	rm -f zen/algorithms/community/*.c zen/algorithms/community/*.so zen/algorithms/community/*.pyc; \
	rm -f zen/drawing/*.c zen/drawing/*.so zen/drawing/*.pyc; \
	rm -f zen/generating/*.c zen/generating/*.so zen/generating/*.pyc; \
	rm -f zen/io/*.c zen/io/*.so zen/io/*.pyc; \
	rm -f zen/layout/*.c zen/layout/*.so zen/layout/*.pyc; \
	rm -f zen/util/fiboheap.c zen/util/queue.c zen/util/*.so zen/util/*.pyc;