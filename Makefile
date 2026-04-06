VENV_DIR   := .venv
BUILD_DIR  := build
PYTHON_VER := 3.14
NPROC      := $(shell nproc 2>/dev/null || echo 4)
UV         := uv

.PHONY: all venv build install test test-c test-py clean distclean help

all: install

# Virtual environment
venv: $(VENV_DIR)/pyvenv.cfg

$(VENV_DIR)/pyvenv.cfg:
	$(UV) venv --python $(PYTHON_VER) $(VENV_DIR)

# Install pybind11 so cmake can find it
$(VENV_DIR)/.pybind11_installed: $(VENV_DIR)/pyvenv.cfg
	$(UV) pip install pybind11
	@touch $@

# Build C library + Python extension, install .so files into curfp/
$(BUILD_DIR)/libcurfp.so: $(wildcard src/*.cpp) $(wildcard include/*.h) CMakeLists.txt \
                           $(VENV_DIR)/.pybind11_installed
	mkdir -p $(BUILD_DIR)
	rm -rf $(BUILD_DIR)/CMakeCache.txt $(BUILD_DIR)/CMakeFiles
	cd $(BUILD_DIR) && cmake .. \
	    -G "Unix Makefiles" \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCURFP_BUILD_PYTHON=ON \
	    -DPYTHON_EXECUTABLE=$(CURDIR)/$(VENV_DIR)/bin/python \
	    -Dpybind11_DIR=$$($(CURDIR)/$(VENV_DIR)/bin/python -m pybind11 --cmakedir)
	$(MAKE) -C $(BUILD_DIR) -j$(NPROC)
	cmake --install $(BUILD_DIR) --prefix $(CURDIR)

build: $(BUILD_DIR)/libcurfp.so

# Install torch, run cmake build, then pip install -e . (pure Python editable)
$(VENV_DIR)/.torch_installed: $(VENV_DIR)/pyvenv.cfg
	$(UV) pip install torch
	@touch $@

install: $(VENV_DIR)/.torch_installed $(BUILD_DIR)/libcurfp.so
	$(UV) pip install -e .

# Tests
test-c: $(BUILD_DIR)/libcurfp.so
	cd $(BUILD_DIR) && ctest --output-on-failure

test-py: install
	$(UV) run python python_tests/test_ssfrk.py
	$(UV) run python python_tests/test_spftrf.py

test: test-c test-py

# Cleanup
clean:
	rm -rf $(BUILD_DIR)
	rm -rf CMakeCache.txt CMakeFiles  # stray root-level cmake artifacts
	find . -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true
	find curfp -name "*.so"    -delete 2>/dev/null || true
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

distclean: clean
	rm -rf $(VENV_DIR)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "  all         Create venv, build, install (default)"
	@echo "  venv        Create .venv with Python $(PYTHON_VER)"
	@echo "  build       cmake build + install .so files into curfp/"
	@echo "  install     build + pip install -e . (pure Python editable)"
	@echo "  test-c      Run C++ tests (ctest)"
	@echo "  test-py     Run Python tests"
	@echo "  test        Run all tests"
	@echo "  clean       Remove build/, compiled .so files"
	@echo "  distclean   clean + remove .venv"
	@echo ""
	@echo "  PYTHON_VER=$(PYTHON_VER)  (override: make PYTHON_VER=3.12)"
