all: update_zen build_zen run_benchmarks upload_figures

update_zen:
	rm -rf zenlib/src; \
	git pull zenlib

build_zen:
	cd ../../src; \
	python setup.py build_ext --inplace

run_benchmarks:
	export PYTHONPATH="../../src:${PYTHONPATH}"; \
	python -m zen.benchmark plots

upload:
	scp *.png druths.webfactional.com:~/webapps/rr_static/zen/benchmarks
