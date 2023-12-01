export HALIDE_ROOT="/home/${USER}/Halide"
export LD_LIBRARY_PATH="${HALIDE_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin:${PATH}


rm -rf build
mkdir build
cmake -S . -B build
cd build
make