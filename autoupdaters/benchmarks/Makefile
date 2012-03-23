all: update_zen build_zen run_benchmarks upload_figures

setup_benchmark_dir:
	svn co https://zenlib.googlecode.com/svn/trunk/ zenlib

update_zen:
	rm -rf zenlib/src; \
	svn update zenlib

build_zen:
	cd zenlib/src; \
	python setup.py build_ext --inplace

run_benchmarks:
	export PYTHONPATH="zenlib/src:${PYTHONPATH}"; \
	python -m zen.benchmark plots

upload_figures:
	scp *.png druths.webfactional.com:~/webapps/rr_static/zen/benchmarks
