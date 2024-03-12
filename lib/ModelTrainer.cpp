// ModelTrainer.cpp

#include "ModelTrainer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/signature_runner.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

const int BATCH_SIZE = 32;
const int NUM_LABELS = 10;
const int IMG_SIZE = 28 * 28;

ModelTrainer::ModelTrainer(const std::string& modelPath) : modelPath(modelPath) {

  // 1. Загружаем модель из файла.
  this->model = loadModel();

  if (this->model) {

    // 2. Создаем Interpreter.
    this->interpreter = makeInterpreter();
  }
}

// Метод обучения модели.
void ModelTrainer::trainModel() {
    if (!interpreter) {
        std::cerr << "Интерпретатор не был инициализирован." << std::endl;
        return;
    }

    // 1. Проверяем точность до начала обучения.
    float beforeTrainingAccuracy = runTest();
    std::cout << "Точность до начала обучения: " << beforeTrainingAccuracy << "%" << std::endl;

    // 2. Инициализируем начальные веса.

    // 3. Запускаем обучение модели.
    runTraining();

    // 4. Проверяем точность после обучения.
    float postTrainingAccuracy = runTest();
    std::cout << "Точность после обучения: " << postTrainingAccuracy << "%" << std::endl;

    // 5. Получаем/сохраняем градиенты у модели.
    tflite::SignatureRunner* save_runner = interpreter.get()->GetSignatureRunner("save");
    auto str = "./checkpoint";
    auto path = save_runner->input_tensor("checkpoint_path");

    if (save_runner->Invoke() != kTfLiteOk) {
      std::cout << "Error invoking save_runner interpreter signature" << std::endl;
    }

    // fill_tensor_str https://github.com/islet-project/islet/blob/34382ad329862bd1625eaae5b20739f062f321cf/examples/confidential-ml/common/word_model.cc#L36
    tflite::DynamicBuffer buf;
    buf.AddString(str, strlen(str));
    buf.WriteToTensorAsVector(path);
}

// export LD_LIBRARY_PATH=/data/local/tmp:${LD_LIBRARY_PATH}

// Метод загрузки модели формата tflite.
std::unique_ptr<tflite::FlatBufferModel> ModelTrainer::loadModel() const {
    auto model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());

    if (!model) {
        std::cerr << "Не удалось загрузить модель по переданному пути: " << modelPath << std::endl;
    } else {
        std::cout << "Модель успешно загружена по переданному пути: " << modelPath << std::endl;
    }

    return model;
}

// Метод создания интерпретатора.
std::unique_ptr<tflite::Interpreter> ModelTrainer::makeInterpreter() const {
    if (!model) {
        return nullptr;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    if (interpreter && interpreter->AllocateTensors() == kTfLiteOk) {
        std::cout << "Интерпретатор создан и тензоры успешно распределены." << std::endl;
    } else {
        std::cerr << "Не удалось создать или выделить тензоры для интерпретатора." << std::endl;
        interpreter.reset();
    }

    return interpreter;
}

float ModelTrainer::runTest() const {
    // Placeholder for test logic; should be replaced with actual model testing code
    std::cout << "Running tests on the model..." << std::endl;

    tflite::SignatureRunner* inference_runner = interpreter.get()->GetSignatureRunner("infer");

    TfLiteTensor* input_tensor = inference_runner->input_tensor(inference_runner->input_names()[0]);
    float* input = input_tensor->data.f;

    int total = 0;
    int correct = 0;

    std::ifstream test_data("test_set.txt");
    std::string line;
    while (std::getline(test_data, line))
    {
        std::istringstream ss(line);
        int label;
        ss >> label;
        for (std::size_t i = 0; i < IMG_SIZE; ++i)
        {
            ss >> input[i];
        }

        if (inference_runner->Invoke() != kTfLiteOk)
        {
            std::cout << "Error invoking inference interpreter signature" << std::endl;
            return 0.f;
        }

        const TfLiteTensor* output_tensor = inference_runner->output_tensor(inference_runner->output_names()[0]);
        float* output = output_tensor->data.f;

        int predicted = 0;
        for (int i = 1; i < NUM_LABELS; ++i)
        {
            if (output[i] > output[predicted])
            {
                predicted = i;
            }
        }

        ++total;
        if (label == predicted)
        {
            ++correct;
        }
    }

    float accuracy = correct * 100.f / total;

    return accuracy;
}

// Функция запуска обучения
void ModelTrainer::runTraining() {
  std::cout << "Запускаем обучение модели..." << std::endl;

  auto train_runner = interpreter.get()->GetSignatureRunner("train");

  TfLiteTensor* input_data_tensor = train_runner->input_tensor(train_runner->input_names()[0]);
  float* input_data = input_data_tensor->data.f;
  TfLiteTensor* input_labels_tensor = train_runner->input_tensor(train_runner->input_names()[1]);
  float* input_labels = input_labels_tensor->data.f;

  int data_idx = 0;
  int num_batches = 0;
  std::vector<float> test_accs;

  std::ifstream train_data("train_set.txt"); // скорее всего это значение интенсивностей пикселей
  std::string line;
  while (std::getline(train_data, line))
  {
    std::istringstream ss(line);
    int label;
    ss >> label;
    for (int lbl_idx = 0; lbl_idx < NUM_LABELS; ++lbl_idx)
    {
      input_labels[data_idx * NUM_LABELS + lbl_idx] = (label == lbl_idx ? 1.f : 0.f);
    }
    for (int i = 0; i < IMG_SIZE; ++i)
    {
      ss >> input_data[data_idx * IMG_SIZE + i];
    }
    ++data_idx;

        if (data_idx == BATCH_SIZE)
        {
            data_idx = 0;

            if (train_runner->Invoke() != kTfLiteOk)
            {
                std::cout << "Error invoking train interpreter signature" << std::endl;
                return;
            }

            ++num_batches;
            const TfLiteTensor* output_tensor = train_runner->output_tensor(train_runner->output_names()[0]);
            float* output = output_tensor->data.f;
            std::cout << "Training of batch " << num_batches << " finished with loss: " << output[0] << std::endl;

            if (num_batches == 100)
            {
                break;
            }
        }
    }
}