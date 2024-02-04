#!/bin/sh
set -eux

PRJ_ROOT="$( cd "$(dirname "$0")" ; pwd -P)"
BUILD_ROOT=$PRJ_ROOT/build_android_arm64

# Check ANDROID NDK
if [ -z "$ANDROID_NDK" ]; then
  echo "ANDROID_NDK not set; please set it to the Android NDK directory"
  exit 1
fi

# Build demo project
rm -rf $BUILD_ROOT && mkdir -p $BUILD_ROOT && cd $BUILD_ROOT

cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=install \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DANDROID_ABI=arm64-v8a \
-DANDROID_NATIVE_API_LEVEL=26 \

make

echo "Build succeeded!"
echo
echo "Run binary on Android:"
echo "adb push build_android/Predictor mobilenetv2.pt /data/local/tmp"
echo "adb shell 'cd /data/local/tmp; ./Predictor mobilenetv2.pt'"
