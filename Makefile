SETUP_BASE=meson setup -Dtest=true --wrap-mode=forcefallback

clear:
	rm -rf build
setup-opt: clear
	$(SETUP_BASE) --warnlevel 3 --buildtype=release -Db_ndebug=true build
setup-opt-debug: clear
	$(SETUP_BASE) --warnlevel 3 --optimization=3 -Ddebug=true build
setup-debug: clear
	$(SETUP_BASE) --warnlevel 3 --buildtype=debug build
