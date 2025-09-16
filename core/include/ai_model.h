#pragma once

#include "types.h"
#include <torch/torch.h>
#include <torch/script.h>

namespace trackpilot {

/**
 * @brief AI Model wrapper for PyTorch TorchScript models
 * 
 * Handles loading and inference of trained PyTorch models for train scheduling.
 * Supports both CPU and GPU inference with batch processing capabilities.
 */
class AIModel {
public:
    /**
     * @brief Constructor
     * @param config Model configuration parameters
     */
    explicit AIModel(const ModelConfig& config);
    
    /**
     * @brief Destructor
     */
    ~AIModel();
    
    /**
     * @brief Load the TorchScript model from file
     * @param model_path Path to the .pt model file
     * @return true if successfully loaded, false otherwise
     */
    bool loadModel(const std::string& model_path);
    
    /**
     * @brief Check if model is loaded and ready
     * @return true if model is ready for inference
     */
    bool isReady() const;
    
    /**
     * @brief Generate scheduling recommendations
     * @param trains List of trains to schedule
     * @param sections Available railway sections
     * @param current_time Current system time
     * @return AI-generated scheduling recommendation
     */
    AIRecommendation generateRecommendation(
        const std::vector<TrainInfo>& trains,
        const std::vector<SectionInfo>& sections,
        const TimePoint& current_time
    );
    
    /**
     * @brief Predict train delay probability
     * @param train Train information
     * @param weather_conditions Weather data (encoded)
     * @param traffic_density Current traffic density
     * @return Probability of delay (0.0-1.0)
     */
    double predictDelayProbability(
        const TrainInfo& train,
        const std::vector<double>& weather_conditions,
        double traffic_density
    );
    
    /**
     * @brief Update model with override feedback
     * @param override_data Human operator override information
     */
    void updateFromOverride(const HumanOverride& override_data);
    
    /**
     * @brief Get model performance metrics
     * @return Map of metric names to values
     */
    std::unordered_map<std::string, double> getMetrics() const;
    
    /**
     * @brief Set device for inference
     * @param device "cpu" or "cuda"
     */
    void setDevice(const std::string& device);

private:
    /**
     * @brief Preprocess input data for the model
     * @param trains Train data
     * @param sections Section data
     * @return Preprocessed tensor
     */
    torch::Tensor preprocessInput(
        const std::vector<TrainInfo>& trains,
        const std::vector<SectionInfo>& sections
    );
    
    /**
     * @brief Postprocess model output to recommendation
     * @param output Model output tensor
     * @param trains Original train data
     * @param sections Original section data
     * @return Structured recommendation
     */
    AIRecommendation postprocessOutput(
        const torch::Tensor& output,
        const std::vector<TrainInfo>& trains,
        const std::vector<SectionInfo>& sections
    );
    
    /**
     * @brief Calculate confidence score for recommendation
     * @param output Model output tensor
     * @return Confidence score (0.0-1.0)
     */
    double calculateConfidence(const torch::Tensor& output);
    
    /**
     * @brief Generate explanation for the recommendation
     * @param recommendation Generated recommendation
     * @param trains Train data
     * @return Human-readable explanation
     */
    std::string generateExplanation(
        const AIRecommendation& recommendation,
        const std::vector<TrainInfo>& trains
    );

    // Member variables
    ModelConfig config_;
    torch::jit::script::Module model_;
    torch::Device device_;
    bool is_loaded_;
    std::unordered_map<std::string, double> metrics_;
    
    // Model statistics
    size_t inference_count_;
    double avg_inference_time_ms_;
    double accuracy_score_;
};

} // namespace trackpilot