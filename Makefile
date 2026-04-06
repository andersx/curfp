VENV_DIR   := .venv
BUILD_DIR  := build
PYTHON_VER := 3.14
NPROC      := $(shell nproc 2>/dev/null || echo 4)
UV         := uv

.PHONY: all venv install test test-c test-py clean distclean help

all: install

# Create virtual environment
venv: $(VENV_DIR)/pyvenv.cfg

$(VENV_DIR)/pyvenv.cfg:
	$(UV) venv --python $(PYTHON_VER) $(VENV_DIR)

# Install the package into the venv (builds C library + Python extension via scikit-build-core)
install: venv
	$(UV) pip install -e .

# C++ tests only (requires a prior cmake build in BUILD_DIR)
test-c: $(BUILD_DIR)/libcurfp.so
	cd $(BUILD_DIR) && ctest --output-on-failure

# Python tests
test-py: install
	$(UV) run pytest python_tests/

test: test-py

# C-only cmake build for running C++ tests without pip
$(BUILD_DIR)/libcurfp.so: $(wildcard src/*.cpp) $(wildcard include/*.h) CMakeLists.txt
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
	$(MAKE) -C $(BUILD_DIR) -j$(NPROC)

# Cleanup
clean:
	rm -rf $(BUILD_DIR)
	find . -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true
	find curfp -name "*.so"    -delete 2>/dev/null || true
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

distclean: clean
	rm -rf $(VENV_DIR)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "  all / install   Create venv and pip install -e . (default)"
	@echo "  venv            Create .venv with Python $(PYTHON_VER)"
	@echo "  test            Run Python tests (pytest)"
	@echo "  test-c          Run C++ tests (ctest)"
	@echo "  clean           Remove build artefacts"
	@echo "  distclean       clean + remove .venv"
	@echo ""
	@echo "  PYTHON_VER=$(PYTHON_VER)  (override: make PYTHON_VER=3.12)"
