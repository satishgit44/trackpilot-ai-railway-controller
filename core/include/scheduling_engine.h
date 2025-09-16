#pragma once

#include "types.h"
#include "ai_model.h"
#include "rule_checker.h"
#include "logger.h"
#include <memory>
#include <queue>
#include <mutex>

namespace trackpilot {

/**
 * @brief Core scheduling engine that combines AI recommendations with safety rules
 * 
 * This class orchestrates the entire scheduling process by:
 * 1. Managing train and section data
 * 2. Calling AI model for recommendations
 * 3. Validating recommendations against safety rules
 * 4. Handling human overrides
 * 5. Maintaining scheduling state
 */
class SchedulingEngine {
public:
    /**
     * @brief Constructor
     * @param config System configuration
     */
    explicit SchedulingEngine(const SystemConfig& config);
    
    /**
     * @brief Destructor
     */
    ~SchedulingEngine();
    
    /**
     * @brief Initialize the scheduling engine
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Add a train to the scheduling system
     * @param train Train information
     */
    void addTrain(const TrainInfo& train);
    
    /**
     * @brief Remove a train from the scheduling system
     * @param train_id Train identifier
     */
    void removeTrain(const std::string& train_id);
    
    /**
     * @brief Update train information
     * @param train Updated train information
     */
    void updateTrain(const TrainInfo& train);
    
    /**
     * @brief Add a railway section
     * @param section Section information
     */
    void addSection(const SectionInfo& section);
    
    /**
     * @brief Generate optimal schedule for all trains
     * @param horizon_hours Scheduling horizon in hours
     * @return AI recommendation with safety validation
     */
    AIRecommendation generateSchedule(int horizon_hours = 24);
    
    /**
     * @brief Apply human operator override
     * @param override Override information from human operator
     * @return Updated schedule after override
     */
    AIRecommendation applyOverride(const HumanOverride& override);
    
    /**
     * @brief Get current schedule
     * @return Current active schedule
     */
    const AIRecommendation& getCurrentSchedule() const;
    
    /**
     * @brief Check if a proposed schedule change is safe
     * @param schedule_change Proposed schedule modifications
     * @return Safety validation result
     */
    std::vector<RuleViolation> validateScheduleChange(
        const std::vector<ScheduleEntry>& schedule_change
    );
    
    /**
     * @brief Get real-time status of all trains
     * @return Map of train_id to current status
     */
    std::unordered_map<std::string, TrainStatus> getTrainStatuses() const;
    
    /**
     * @brief Update real-time train position
     * @param train_id Train identifier
     * @param section_id Current section
     * @param position_km Position within section (km)
     */
    void updateTrainPosition(const std::string& train_id, 
                           const std::string& section_id, 
                           double position_km);
    
    /**
     * @brief Get performance metrics
     * @return System performance metrics
     */
    std::unordered_map<std::string, double> getPerformanceMetrics() const;
    
    /**
     * @brief Enable/disable learning from overrides
     * @param enable True to enable learning
     */
    void setLearningEnabled(bool enable);
    
    /**
     * @brief Get list of all trains
     * @return Vector of train information
     */
    std::vector<TrainInfo> getAllTrains() const;
    
    /**
     * @brief Get list of all sections
     * @return Vector of section information
     */
    std::vector<SectionInfo> getAllSections() const;

private:
    /**
     * @brief Optimize schedule using heuristic algorithms
     * @param recommendation Initial AI recommendation
     * @return Optimized schedule
     */
    AIRecommendation optimizeSchedule(const AIRecommendation& recommendation);
    
    /**
     * @brief Handle conflicts in the schedule
     * @param conflicts List of detected conflicts
     * @return Resolved schedule entries
     */
    std::vector<ScheduleEntry> resolveConflicts(
        const std::vector<RuleViolation>& conflicts
    );
    
    /**
     * @brief Calculate scheduling metrics
     * @param schedule Current schedule
     * @return Performance metrics
     */
    std::unordered_map<std::string, double> calculateMetrics(
        const AIRecommendation& schedule
    ) const;
    
    /**
     * @brief Log override for learning purposes
     * @param override Override information
     */
    void logOverride(const HumanOverride& override);
    
    /**
     * @brief Update AI model with recent overrides
     */
    void updateModelFromOverrides();
    
    /**
     * @brief Check for real-time updates needed
     */
    void checkRealTimeUpdates();

    // Core components
    std::unique_ptr<AIModel> ai_model_;
    std::unique_ptr<RuleChecker> rule_checker_;
    std::unique_ptr<Logger> logger_;
    
    // System configuration
    SystemConfig config_;
    
    // Data storage
    std::unordered_map<std::string, TrainInfo> trains_;
    std::unordered_map<std::string, SectionInfo> sections_;
    AIRecommendation current_schedule_;
    
    // Real-time tracking
    std::unordered_map<std::string, TrainStatus> train_statuses_;
    std::unordered_map<std::string, std::pair<std::string, double>> train_positions_;
    
    // Override management
    std::queue<HumanOverride> override_queue_;
    bool learning_enabled_;
    
    // Thread safety
    mutable std::mutex trains_mutex_;
    mutable std::mutex sections_mutex_;
    mutable std::mutex schedule_mutex_;
    
    // Performance tracking
    std::chrono::time_point<std::chrono::steady_clock> last_update_;
    size_t total_schedules_generated_;
    double avg_generation_time_ms_;
    double schedule_efficiency_score_;
};

} // namespace trackpilot