// ModelTrainer.h

#ifndef MODEL_TRAINER_H
#define MODEL_TRAINER_H

#include <string>
#include <memory>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

class ModelTrainer {
public:
    explicit ModelTrainer(const std::string& modelPath);
    void trainModel();

private:
    const std::string modelPath;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

    std::unique_ptr<tflite::FlatBufferModel> loadModel() const;
    std::unique_ptr<tflite::Interpreter> makeInterpreter() const;
    float runTest() const;
    void runTraining();
};

#endif // MODEL_TRAINER_H
