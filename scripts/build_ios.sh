#!/bin/bash
##############################################################################
# Example command to build the iOS target.
##############################################################################
#
# This script shows how one can build a Caffe2 binary for the iOS platform
# using ios-cmake. This is very similar to the android-cmake - see
# build_android.sh for more details.

set -e

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"

# Build protobuf from third_party so we have a host protoc binary.
echo "Building protoc"
BITCODE_FLAGS="-DCMAKE_C_FLAGS=-fembed-bitcode -DCMAKE_CXX_FLAGS=-fembed-bitcode "
$CAFFE2_ROOT/scripts/build_host_protoc.sh --other-flags $BITCODE_FLAGS

# Now, actually build the iOS target.
BUILD_ROOT=${BUILD_ROOT:-"$CAFFE2_ROOT/build_ios"}
mkdir -p $BUILD_ROOT
cd $BUILD_ROOT

CMAKE_ARGS=()

# Use locally built protoc because we'll build libprotobuf for the
# target architecture and need an exact version match.
CMAKE_ARGS+=("-DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=$CAFFE2_ROOT/build_host_protoc/bin/protoc")

# Use ios-cmake to build iOS project from CMake.
# This projects sets CMAKE_C_COMPILER to /usr/bin/gcc and
# CMAKE_CXX_COMPILER to /usr/bin/g++. In order to use sccache (if it is available) we
# must override these variables via CMake arguments.
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$CAFFE2_ROOT/third_party/ios-cmake/toolchain/iOS.cmake")
if which sccache > /dev/null; then
  for compiler in cc c++ gcc g++ clang clang++; do
    (
      echo "#!/bin/sh"
      echo "exec sccache $(which $compiler) \"\$@\""
    ) > "./sccache/$compiler"
    chmod +x "./sccache/$compiler"
  done
  export CACHE_WRAPPER_DIR="$PWD/sccache"

  CMAKE_ARGS+=("-DCMAKE_C_COMPILER=$CACHE_WRAPPER_DIR/gcc")
  CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER=$CACHE_WRAPPER_DIR/g++")

  sccache --zero-stats
  sccache --show-stats
fi

# IOS_PLATFORM controls type of iOS platform (see ios-cmake)
if [ -n "${IOS_PLATFORM:-}" ]; then
  CMAKE_ARGS+=("-DIOS_PLATFORM=${IOS_PLATFORM}")
else
  # IOS_PLATFORM is not set, default to OS, which builds iOS.
  CMAKE_ARGS+=("-DIOS_PLATFORM=OS")
fi

# Don't build binaries or tests (only the library)
CMAKE_ARGS+=("-DBUILD_TEST=OFF")
CMAKE_ARGS+=("-DBUILD_BINARY=OFF")
CMAKE_ARGS+=("-DBUILD_PYTHON=OFF")

# Disable unused dependencies
CMAKE_ARGS+=("-DUSE_CUDA=OFF")
CMAKE_ARGS+=("-DUSE_GFLAGS=OFF")
CMAKE_ARGS+=("-DUSE_OPENCV=OFF")
CMAKE_ARGS+=("-DUSE_LMDB=OFF")
CMAKE_ARGS+=("-DUSE_LEVELDB=OFF")
CMAKE_ARGS+=("-DUSE_MPI=OFF")

# pthreads
CMAKE_ARGS+=("-DCMAKE_THREAD_LIBS_INIT=-lpthread")
CMAKE_ARGS+=("-DCMAKE_HAVE_THREADS_LIBRARY=1")
CMAKE_ARGS+=("-DCMAKE_USE_PTHREADS_INIT=1")

# Only toggle if VERBOSE=1
if [ "${VERBOSE:-}" == '1' ]; then
  CMAKE_ARGS+=("-DCMAKE_VERBOSE_MAKEFILE=1")
fi

CMAKE_ARGS+=("-DCMAKE_C_FLAGS=-fembed-bitcode")
CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS=-fembed-bitcode")

cmake "$CAFFE2_ROOT" \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    ${CMAKE_ARGS[@]} \
    $@

cmake --build . -- "-j$(sysctl -n hw.ncpu)"

if which sccache > /dev/null; then
  sccache --show-stats
fi
