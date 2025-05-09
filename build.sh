mkdir -p build && cd build
# -DCMAKE_MAKE_PROGRAM=make -DCMAKE_C_COMPILER=/usr/bin/clang-18 -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18
cmake -DCMAKE_TOOLCHAIN_FILE=../clang-toolchain.cmake -S .. -B .
make