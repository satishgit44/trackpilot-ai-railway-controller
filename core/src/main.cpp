#include <iostream>
#include <string>
#include <memory>
#include <csignal>
#include <atomic>

#include "scheduling_engine.h"
#include "api_server.h"
#include "config.h"
#include "logger.h"

namespace {
    std::atomic<bool> running{true};
    std::unique_ptr<trackpilot::APIServer> server;
}

void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\nShutting down TrackPilot Core..." << std::endl;
        running = false;
        if (server) {
            server->stop();
        }
    }
}

void printUsage(const std::string& program_name) {
    std::cout << "TrackPilot AI Railway Controller - Core Engine\n"
              << "Usage: " << program_name << " [options]\n\n"
              << "Options:\n"
              << "  -h, --help              Show this help message\n"
              << "  -c, --config FILE       Configuration file path\n"
              << "  -m, --model FILE        AI model file path\n"
              << "  -p, --port PORT         API server port (default: 8080)\n"
              << "  --host HOST             API server host (default: 0.0.0.0)\n"
              << "  -v, --verbose           Enable verbose logging\n"
              << "  --no-server             Run without HTTP API server\n"
              << "  --cli                   Interactive CLI mode\n"
              << "\nExamples:\n"
              << "  " << program_name << " --config config.json --model model.pt\n"
              << "  " << program_name << " --port 9090 --verbose\n"
              << "  " << program_name << " --cli\n\n";
}

int runCLIMode(trackpilot::SchedulingEngine& engine) {
    std::cout << "TrackPilot Interactive CLI Mode\n";
    std::cout << "Type 'help' for available commands, 'quit' to exit.\n\n";
    
    std::string input;
    while (running && std::cout << "trackpilot> " && std::getline(std::cin, input)) {
        if (input == "quit" || input == "exit") {
            break;
        } else if (input == "help") {
            std::cout << "Available commands:\n"
                      << "  status                 - Show system status\n"
                      << "  schedule [hours]       - Generate schedule (default 24 hours)\n"
                      << "  trains                 - List all trains\n"
                      << "  sections               - List all sections\n"
                      << "  metrics                - Show performance metrics\n"
                      << "  add_train              - Add a new train (interactive)\n"
                      << "  help                   - Show this help\n"
                      << "  quit                   - Exit CLI\n\n";
        } else if (input == "status") {
            auto train_statuses = engine.getTrainStatuses();
            std::cout << "System Status:\n";
            std::cout << "Active trains: " << train_statuses.size() << "\n";
            std::cout << "Current schedule confidence: " << 
                engine.getCurrentSchedule().overall_confidence << "\n\n";
        } else if (input.substr(0, 8) == "schedule") {
            int hours = 24;
            if (input.length() > 9) {
                try {
                    hours = std::stoi(input.substr(9));
                } catch (...) {
                    std::cout << "Invalid hour format. Using default 24 hours.\n";
                }
            }
            
            auto recommendation = engine.generateSchedule(hours);
            std::cout << "Generated schedule for " << hours << " hours:\n";
            std::cout << "Confidence: " << recommendation.overall_confidence << "\n";
            std::cout << "Entries: " << recommendation.schedule.size() << "\n";
            std::cout << "Reasoning: " << recommendation.reasoning << "\n\n";
        } else if (input == "trains") {
            auto trains = engine.getAllTrains();
            std::cout << "Trains (" << trains.size() << "):\n";
            for (const auto& train : trains) {
                std::cout << "  " << train.train_id << " - " << train.route 
                          << " (Priority: " << train.priority << ")\n";
            }
            std::cout << "\n";
        } else if (input == "sections") {
            auto sections = engine.getAllSections();
            std::cout << "Sections (" << sections.size() << "):\n";
            for (const auto& section : sections) {
                std::cout << "  " << section.section_id << " - " << section.length_km 
                          << " km (Max trains: " << section.max_trains << ")\n";
            }
            std::cout << "\n";
        } else if (input == "metrics") {
            auto metrics = engine.getPerformanceMetrics();
            std::cout << "Performance Metrics:\n";
            for (const auto& metric : metrics) {
                std::cout << "  " << metric.first << ": " << metric.second << "\n";
            }
            std::cout << "\n";
        } else if (input == "add_train") {
            // Interactive train addition
            std::cout << "Adding new train (press Enter to skip optional fields):\n";
            std::string train_id, route;
            int priority = 5;
            double speed = 80.0;
            
            std::cout << "Train ID: ";
            std::getline(std::cin, train_id);
            std::cout << "Route: ";
            std::getline(std::cin, route);
            std::cout << "Priority (1-10, default 5): ";
            std::string priority_str;
            std::getline(std::cin, priority_str);
            if (!priority_str.empty()) {
                try {
                    priority = std::stoi(priority_str);
                } catch (...) {}
            }
            
            trackpilot::TrainInfo train;
            train.train_id = train_id;
            train.route = route;
            train.priority = priority;
            train.speed_kmh = speed;
            train.is_freight = false;
            train.scheduled_arrival = std::chrono::system_clock::now();
            train.scheduled_departure = std::chrono::system_clock::now() + std::chrono::hours(1);
            
            engine.addTrain(train);
            std::cout << "Train " << train_id << " added successfully.\n\n";
        } else if (!input.empty()) {
            std::cout << "Unknown command: " << input << "\nType 'help' for available commands.\n\n";
        }
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    // Install signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Parse command line arguments
    std::string config_file = "config.json";
    std::string model_file;
    std::string host = "0.0.0.0";
    int port = 8080;
    bool verbose = false;
    bool no_server = false;
    bool cli_mode = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                config_file = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires a file path\n";
                return 1;
            }
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                model_file = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires a file path\n";
                return 1;
            }
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) {
                port = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: " << arg << " requires a port number\n";
                return 1;
            }
        } else if (arg == "--host") {
            if (i + 1 < argc) {
                host = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires a host address\n";
                return 1;
            }
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--no-server") {
            no_server = true;
        } else if (arg == "--cli") {
            cli_mode = true;
            no_server = true;
        } else {
            std::cerr << "Error: Unknown option " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    
    try {
        // Load configuration
        auto config = trackpilot::Config::loadFromFile(config_file);
        
        // Override with command line arguments
        if (!model_file.empty()) {
            config.model.model_path = model_file;
        }
        config.server.host = host;
        config.server.port = port;
        if (verbose) {
            config.server.log_level = "DEBUG";
        }
        
        std::cout << "TrackPilot AI Railway Controller - Core Engine v1.0\n";
        std::cout << "Configuration: " << config_file << "\n";
        if (!config.model.model_path.empty()) {
            std::cout << "AI Model: " << config.model.model_path << "\n";
        }
        std::cout << "Log Level: " << config.server.log_level << "\n\n";
        
        // Initialize scheduling engine
        trackpilot::SchedulingEngine engine(config);
        if (!engine.initialize()) {
            std::cerr << "Failed to initialize scheduling engine\n";
            return 1;
        }
        
        std::cout << "Scheduling engine initialized successfully\n";
        
        if (cli_mode) {
            return runCLIMode(engine);
        }
        
        if (!no_server) {
            // Start API server
            server = std::make_unique<trackpilot::APIServer>(config.server, engine);
            if (!server->start()) {
                std::cerr << "Failed to start API server\n";
                return 1;
            }
            
            std::cout << "API server running on " << host << ":" << port << "\n";
            std::cout << "Press Ctrl+C to stop...\n\n";
            
            // Keep running until signal received
            while (running) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            
            server->stop();
        } else {
            std::cout << "Running in daemon mode (no API server)\n";
            std::cout << "Press Ctrl+C to stop...\n\n";
            
            while (running) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        
        std::cout << "TrackPilot Core Engine stopped\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred\n";
        return 1;
    }
}