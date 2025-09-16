#pragma once

#include "types.h"
#include <functional>
#include <unordered_set>

namespace trackpilot {

/**
 * @brief Safety rule checker for railway scheduling validation
 * 
 * Implements comprehensive safety rules for railway operations including:
 * - Signal spacing and timing constraints
 * - Track capacity limitations
 * - Speed and braking distance calculations
 * - Priority and conflict resolution rules
 */
class RuleChecker {
public:
    /**
     * @brief Constructor
     */
    RuleChecker();
    
    /**
     * @brief Destructor
     */
    ~RuleChecker();
    
    /**
     * @brief Initialize rule checker with safety rules
     * @param rules_file_path Path to safety rules configuration file
     * @return true if initialization successful
     */
    bool initialize(const std::string& rules_file_path = "");
    
    /**
     * @brief Validate a complete schedule against all safety rules
     * @param schedule Schedule to validate
     * @param trains Train information
     * @param sections Section information
     * @return List of rule violations (empty if all rules satisfied)
     */
    std::vector<RuleViolation> validateSchedule(
        const std::vector<ScheduleEntry>& schedule,
        const std::vector<TrainInfo>& trains,
        const std::vector<SectionInfo>& sections
    );
    
    /**
     * @brief Validate a single schedule change
     * @param entry Schedule entry to validate
     * @param existing_schedule Current schedule
     * @param trains Train information
     * @param sections Section information
     * @return List of violations for this entry
     */
    std::vector<RuleViolation> validateScheduleEntry(
        const ScheduleEntry& entry,
        const std::vector<ScheduleEntry>& existing_schedule,
        const std::vector<TrainInfo>& trains,
        const std::vector<SectionInfo>& sections
    );
    
    /**
     * @brief Check minimum separation between trains
     * @param train1 First train schedule entry
     * @param train2 Second train schedule entry
     * @param section Section information
     * @return true if minimum separation is maintained
     */
    bool checkMinimumSeparation(
        const ScheduleEntry& train1,
        const ScheduleEntry& train2,
        const SectionInfo& section
    );
    
    /**
     * @brief Check track capacity constraints
     * @param schedule_entries All entries for a specific section
     * @param section Section information
     * @return true if capacity not exceeded
     */
    bool checkTrackCapacity(
        const std::vector<ScheduleEntry>& schedule_entries,
        const SectionInfo& section
    );
    
    /**
     * @brief Validate signal timing constraints
     * @param entry Schedule entry to check
     * @param section Section information
     * @return true if signal constraints satisfied
     */
    bool checkSignalConstraints(
        const ScheduleEntry& entry,
        const SectionInfo& section
    );
    
    /**
     * @brief Check priority rules and conflicts
     * @param high_priority_entry Higher priority train
     * @param low_priority_entry Lower priority train
     * @param trains Train information map
     * @return true if priority rules respected
     */
    bool checkPriorityRules(
        const ScheduleEntry& high_priority_entry,
        const ScheduleEntry& low_priority_entry,
        const std::unordered_map<std::string, TrainInfo>& trains
    );
    
    /**
     * @brief Calculate safe braking distance
     * @param speed_kmh Train speed in km/h
     * @param track_condition Track condition factor (0.0-1.0)
     * @return Braking distance in meters
     */
    double calculateBrakingDistance(double speed_kmh, double track_condition = 1.0);
    
    /**
     * @brief Get all active safety rules
     * @return Vector of safety rules
     */
    std::vector<SafetyRule> getAllRules() const;
    
    /**
     * @brief Add custom safety rule
     * @param rule Safety rule to add
     */
    void addRule(const SafetyRule& rule);
    
    /**
     * @brief Remove safety rule by ID
     * @param rule_id Rule identifier
     */
    void removeRule(const std::string& rule_id);
    
    /**
     * @brief Enable or disable a specific rule
     * @param rule_id Rule identifier
     * @param enabled True to enable, false to disable
     */
    void setRuleEnabled(const std::string& rule_id, bool enabled);
    
    /**
     * @brief Get rule checker statistics
     * @return Map of statistic names to values
     */
    std::unordered_map<std::string, double> getStatistics() const;

private:
    /**
     * @brief Load default safety rules
     */
    void loadDefaultRules();
    
    /**
     * @brief Load safety rules from configuration file
     * @param file_path Path to rules file
     * @return true if loaded successfully
     */
    bool loadRulesFromFile(const std::string& file_path);
    
    /**
     * @brief Check temporal overlap between two schedule entries
     * @param entry1 First schedule entry
     * @param entry2 Second schedule entry
     * @param margin_minutes Safety margin in minutes
     * @return true if entries overlap
     */
    bool checkTemporalOverlap(
        const ScheduleEntry& entry1,
        const ScheduleEntry& entry2,
        double margin_minutes = 5.0
    );
    
    /**
     * @brief Validate single track section constraints
     * @param entries All entries for single track section
     * @param section Section information
     * @return List of violations
     */
    std::vector<RuleViolation> validateSingleTrackSection(
        const std::vector<ScheduleEntry>& entries,
        const SectionInfo& section
    );
    
    /**
     * @brief Check freight vs passenger train conflicts
     * @param freight_entry Freight train entry
     * @param passenger_entry Passenger train entry
     * @param trains Train information
     * @return true if no conflicts
     */
    bool checkFreightPassengerConflict(
        const ScheduleEntry& freight_entry,
        const ScheduleEntry& passenger_entry,
        const std::unordered_map<std::string, TrainInfo>& trains
    );
    
    /**
     * @brief Validate speed limits and restrictions
     * @param entry Schedule entry
     * @param train Train information
     * @param section Section information
     * @return List of violations
     */
    std::vector<RuleViolation> validateSpeedLimits(
        const ScheduleEntry& entry,
        const TrainInfo& train,
        const SectionInfo& section
    );

    // Rule storage and management
    std::vector<SafetyRule> rules_;
    std::unordered_set<std::string> disabled_rules_;
    
    // Rule checking functions
    std::unordered_map<std::string, std::function<std::vector<RuleViolation>(
        const std::vector<ScheduleEntry>&,
        const std::vector<TrainInfo>&,
        const std::vector<SectionInfo>&
    )>> rule_functions_;
    
    // Configuration parameters
    double minimum_separation_minutes_;
    double signal_clearance_time_seconds_;
    double emergency_brake_factor_;
    double track_condition_factor_;
    
    // Statistics
    mutable size_t total_validations_;
    mutable size_t total_violations_;
    mutable double avg_validation_time_ms_;
};

} // namespace trackpilot