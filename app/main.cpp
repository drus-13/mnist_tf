// main.cpp

#include <iostream>
#include "ModelTrainer.h"
// #include <boinc/boinc_api.h>

int main() {
    ModelTrainer trainer("./model.tflite");
    trainer.trainModel();
    // boinc_init();
    std::cout << "Training completed." << std::endl;
    return 0;
}
