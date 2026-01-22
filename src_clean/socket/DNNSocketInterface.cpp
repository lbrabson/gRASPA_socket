/**
 * @file DNNSocketInterface.cpp
 * @brief Implementation of thread-safe singleton interface for socket DNN calculations
 */

#include "socket/DNNSocketInterface.h"
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace graspa {
namespace socket {

DNNSocketInterface& DNNSocketInterface::getInstance() {
    static DNNSocketInterface instance;
    return instance;
}

DNNSocketInterface::DNNSocketInterface()
    : config_()
    , client_(nullptr)
    , protocol_(nullptr)
    , hostInitialized_(false)
    , hostCell_(9, 0.0)
    , calcCount_(0)
{
}

DNNSocketInterface::~DNNSocketInterface() {
    disconnect();
}

void DNNSocketInterface::configure(const SocketConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (client_ && client_->isConnected()) {
        throw std::runtime_error("Cannot reconfigure while connected");
    }

    config_ = config;
}

bool DNNSocketInterface::isConnected() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return client_ && client_->isConnected();
}

void DNNSocketInterface::connect() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!config_.enabled) {
        throw std::runtime_error("Socket calculator not enabled");
    }

    if (client_ && client_->isConnected()) {
        return;  // Already connected
    }

    // Create new client
    client_ = std::make_unique<SocketClient>();

    // Connect based on type
    if (config_.type == SocketType::Unix) {
        std::cout << "[Socket] Connecting to Unix socket: " << config_.address << std::endl;
        client_->connectUnix(config_.address);
    } else {
        std::cout << "[Socket] Connecting to TCP: " << config_.host << ":" << config_.port << std::endl;
        client_->connectTCP(config_.host, config_.port);
    }

    // Create protocol handler
    protocol_ = std::make_unique<IPIProtocol>(*client_);

    std::cout << "[Socket] Connected successfully" << std::endl;

    hostInitialized_ = false;
    calcCount_ = 0;
}

void DNNSocketInterface::disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (protocol_) {
        try {
            protocol_->sendExit();
        } catch (...) {
            // Ignore errors during disconnect
        }
        protocol_.reset();
    }

    if (client_) {
        client_->close();
        client_.reset();
    }

    hostInitialized_ = false;

    std::cout << "[Socket] Disconnected (total calculations: " << calcCount_ << ")" << std::endl;
}

void DNNSocketInterface::initializeHost(
    const std::vector<double>& cell,
    const std::vector<double>& positions,
    const std::vector<int32_t>& types,
    const std::vector<double>& charges,
    const std::map<int32_t, std::string>& typeMap
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!client_ || !client_->isConnected()) {
        throw std::runtime_error("Not connected to server");
    }

    // Build host data structure
    HostData hostData;
    hostData.cell = cell;
    hostData.positions = positions;
    hostData.types = types;
    hostData.charges = charges;
    hostData.typeMap = typeMap;

    // Cache cell for later use
    hostCell_ = cell;

    // Initialize protocol
    protocol_->initialize(hostData);

    hostInitialized_ = true;

    std::cout << "[Socket] Host initialized with " << types.size() << " atoms, "
              << typeMap.size() << " types" << std::endl;
}

double DNNSocketInterface::calculateGuestEnergy(
    const std::vector<double>& positions,
    const std::vector<int32_t>& types,
    const std::vector<double>& cell
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!client_ || !client_->isConnected()) {
        throw std::runtime_error("Not connected to server");
    }

    if (!hostInitialized_) {
        throw std::runtime_error("Host data not initialized");
    }

    // Build position data structure
    PosData posData;
    posData.cell = cell;
    posData.positions = positions;
    posData.types = types;

    // Calculate energy
    double energy = protocol_->calculateEnergy(posData);

    ++calcCount_;

    return energy;
}

double DNNSocketInterface::calculateGuestEnergy(
    const std::vector<double>& positions,
    const std::vector<int32_t>& types
) {
    // Use cached host cell
    return calculateGuestEnergy(positions, types, hostCell_);
}

SocketConfig parseSocketConfig(const std::string& typeStr, const std::string& address) {
    SocketConfig config;
    config.enabled = true;
    config.address = address;

    std::string typeLower = typeStr;
    for (auto& c : typeLower) {
        c = std::tolower(c);
    }

    if (typeLower == "unix") {
        config.type = SocketType::Unix;
    } else if (typeLower == "tcp") {
        config.type = SocketType::TCP;

        // Parse host:port
        size_t colonPos = address.rfind(':');
        if (colonPos == std::string::npos) {
            throw std::invalid_argument("TCP address must be 'host:port', got: " + address);
        }

        config.host = address.substr(0, colonPos);
        std::string portStr = address.substr(colonPos + 1);

        try {
            config.port = std::stoi(portStr);
        } catch (...) {
            throw std::invalid_argument("Invalid port number: " + portStr);
        }

        if (config.port <= 0 || config.port > 65535) {
            throw std::invalid_argument("Port out of range: " + portStr);
        }
    } else {
        throw std::invalid_argument("Unknown socket type: " + typeStr + " (use 'unix' or 'tcp')");
    }

    return config;
}

} // namespace socket
} // namespace graspa
