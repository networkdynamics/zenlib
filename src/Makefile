
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

test_inplace:
	export PYTHONPATH=.:${PYTHONPATH}; python -m zen.test

benchmark_inplace:
	export PYTHONPATH=.:${PYTHONPATH}; python -m zen.benchmark

clean:
	rm -rf build; \
	rm -f zen/*.c zen/*.so; \
	rm -f zen/community/*.c zen/community/*.so; \
	rm -f zen/drawing/*.c zen/drawing/*.so; \
	rm -f zen/generating/*.c zen/generating/*.so; \
	rm -f zen/io/*.c zen/io/*.so; \
	rm -f zen/layout/*.c zen/layout/*.so; \
	rm -f zen/util/fiboheap.c zen/util/queue.c zen/util/*.so;