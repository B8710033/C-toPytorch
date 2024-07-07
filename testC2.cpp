#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        Ort::Session session(env, L"simple_model.onnx", session_options);

        std::vector<float> input_tensor_values = { 10.0 };
        std::vector<int64_t> input_tensor_shape = { 1 };

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());

        std::vector<Ort::Value> output_tensors;
        output_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size()));

        const char* input_names[] = { "onnx::Add_0" };
        const char* output_names[] = { "2" };

        session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, output_tensors.data(), 1);

        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        std::cout << "Model output: " << floatarr[0] << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
