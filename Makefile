simd_backend=auto
SETUP_BASE=meson setup -Dbuild_tests=true -Dsimd_backend=$(simd_backend) --wrap-mode=forcefallback

clear:
	rm -rf build
setup-opt: clear
	$(SETUP_BASE) --warnlevel 3 --buildtype=release -Db_ndebug=true build
setup-opt-debug: clear
	$(SETUP_BASE) --warnlevel 3 --optimization=3 -Ddebug=true build
setup-debug: clear
	$(SETUP_BASE) --warnlevel 3 --buildtype=debug build
doxy:
	DOXY_INC=include DOXY_OUT=build/doxy doxygen docs/Doxyfile
sphinx: doxy
	sphinx-build -b html -Dbreathe_projects.grex=../build/doxy docs build/sphinx
	python3 docs/fix_sphinx.py build/sphinx
