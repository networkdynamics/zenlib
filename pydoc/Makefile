
build_zen_inplace:
	cd ../src; python setup.py build_ext --inplace

build_pydoc: build_zen_inplace
	export PYTHONPATH="../src:$PYTHONPATH"; epydoc --html -o ../pydoc/autogen --name Zen --url http://www.networkdynamics.org/static/zen zen

publish_pydoc:
	scp -r autogen druths.webfactional.com:~/webapps/rr_static/zen