#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <unordered_map>

namespace trackpilot {

// Time representation
using TimePoint = std::chrono::system_clock::time_point;
using Duration = std::chrono::minutes;

// Basic data structures
struct TrainInfo {
    std::string train_id;
    std::string route;
    TimePoint scheduled_arrival;
    TimePoint scheduled_departure;
    int priority; // 1-10, higher is more important
    double speed_kmh;
    bool is_freight;
    std::vector<std::string> stops;
};

struct SectionInfo {
    std::string section_id;
    double length_km;
    int max_trains;
    bool is_single_track;
    std::vector<std::string> signals;
};

struct ScheduleEntry {
    std::string train_id;
    std::string section_id;
    TimePoint entry_time;
    TimePoint exit_time;
    double confidence_score;
    std::string explanation;
};

struct AIRecommendation {
    std::vector<ScheduleEntry> schedule;
    double overall_confidence;
    std::string reasoning;
    std::unordered_map<std::string, double> metrics;
};

struct HumanOverride {
    TimePoint timestamp;
    std::string operator_id;
    AIRecommendation original_recommendation;
    std::vector<ScheduleEntry> modified_schedule;
    std::string reason;
    std::string feedback;
};

struct SafetyRule {
    std::string rule_id;
    std::string description;
    bool is_mandatory;
    double penalty_score;
};

struct RuleViolation {
    std::string rule_id;
    std::string train_id;
    std::string section_id;
    std::string description;
    double severity; // 0.0-1.0
};

// Configuration structures
struct ModelConfig {
    std::string model_path;
    std::string device; // "cpu" or "cuda"
    int batch_size;
    double confidence_threshold;
};

struct ServerConfig {
    std::string host;
    int port;
    bool enable_cors;
    std::string log_level;
};

struct SystemConfig {
    ModelConfig model;
    ServerConfig server;
    std::string log_directory;
    std::string override_log_path;
    int max_lookahead_hours;
    double safety_margin_minutes;
};

// Enums
enum class TrainStatus {
    SCHEDULED,
    DELAYED,
    ON_TIME,
    CANCELLED,
    DIVERTED
};

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

enum class APIResponseCode {
    SUCCESS = 200,
    BAD_REQUEST = 400,
    NOT_FOUND = 404,
    INTERNAL_ERROR = 500
};

} // namespace trackpilot