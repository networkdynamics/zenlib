
setup_pydoc_dir:
	svn co https://zenlib.googlecode.com/svn/trunk/ zenlib

update_pydoc:
	svn update zenlib; \
	cd zenlib/pydoc; make build_pydoc; make publish_pydoc
