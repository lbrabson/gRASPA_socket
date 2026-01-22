/**
 * @file DNNSocketInterface.h
 * @brief Thread-safe singleton interface for socket-based DNN calculations
 *
 * Provides the main integration point for gRASPA to communicate with
 * the Python ASE calculator server.
 */

#ifndef DNN_SOCKET_INTERFACE_H
#define DNN_SOCKET_INTERFACE_H

#include "socket/SocketClient.h"
#include "socket/IPIProtocol.h"
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <map>

namespace graspa {
namespace socket {

/**
 * @brief Configuration for socket connection
 */
struct SocketConfig {
    bool enabled = false;           ///< Whether socket calculator is enabled
    SocketType type = SocketType::Unix;  ///< Socket type (Unix or TCP)
    std::string address;            ///< Socket path or host:port

    // Parsed TCP fields (populated from address if type is TCP)
    std::string host;
    int port = 0;
};

/**
 * @brief Thread-safe singleton interface for DNN socket calculations
 *
 * This class provides the main integration point for gRASPA to use
 * the socket-based ASE calculator. It handles connection management,
 * thread safety, and provides a simple API for energy calculations.
 *
 * Usage in gRASPA:
 * @code
 * auto& socket = DNNSocketInterface::getInstance();
 * socket.configure(config);
 * socket.connect();
 * socket.initializeHost(hostData);
 *
 * // In MC loop:
 * double energy = socket.calculateGuestEnergy(guestPositions, guestTypes, cell);
 *
 * // Cleanup:
 * socket.disconnect();
 * @endcode
 */
class DNNSocketInterface {
public:
    /**
     * @brief Get the singleton instance
     */
    static DNNSocketInterface& getInstance();

    // Non-copyable, non-movable
    DNNSocketInterface(const DNNSocketInterface&) = delete;
    DNNSocketInterface& operator=(const DNNSocketInterface&) = delete;
    DNNSocketInterface(DNNSocketInterface&&) = delete;
    DNNSocketInterface& operator=(DNNSocketInterface&&) = delete;

    /**
     * @brief Configure the socket connection
     * @param config Socket configuration
     */
    void configure(const SocketConfig& config);

    /**
     * @brief Check if socket calculator is enabled
     */
    bool isEnabled() const { return config_.enabled; }

    /**
     * @brief Check if connected to server
     */
    bool isConnected() const;

    /**
     * @brief Connect to the Python server
     * @throws SocketError on connection failure
     */
    void connect();

    /**
     * @brief Disconnect from the server
     */
    void disconnect();

    /**
     * @brief Initialize with host (framework) atom data
     *
     * Must be called once after connection before any energy calculations.
     *
     * @param cell 3x3 cell matrix (row-major, Angstrom)
     * @param positions Nx3 host positions (flattened, Angstrom)
     * @param types N host atom type IDs
     * @param charges N host partial charges
     * @param typeMap Mapping of type_id -> chemical symbol
     */
    void initializeHost(
        const std::vector<double>& cell,
        const std::vector<double>& positions,
        const std::vector<int32_t>& types,
        const std::vector<double>& charges,
        const std::map<int32_t, std::string>& typeMap
    );

    /**
     * @brief Calculate energy for guest atoms
     *
     * Thread-safe energy calculation. Blocks until energy is returned
     * from the Python server.
     *
     * @param positions Nx3 guest positions (flattened, Angstrom)
     * @param types N guest atom type IDs
     * @param cell 3x3 cell matrix (row-major, Angstrom)
     * @return Total energy in kJ/mol
     * @throws SocketError on communication failure
     */
    double calculateGuestEnergy(
        const std::vector<double>& positions,
        const std::vector<int32_t>& types,
        const std::vector<double>& cell
    );

    /**
     * @brief Simplified energy calculation using current cell
     *
     * Uses the cell from host initialization.
     *
     * @param positions Nx3 guest positions (flattened, Angstrom)
     * @param types N guest atom type IDs
     * @return Total energy in kJ/mol
     */
    double calculateGuestEnergy(
        const std::vector<double>& positions,
        const std::vector<int32_t>& types
    );

    /**
     * @brief Get the number of calculations performed
     */
    size_t calcCount() const { return calcCount_; }

    /**
     * @brief Reset calculation counter
     */
    void resetCalcCount() { calcCount_ = 0; }

    /**
     * @brief Get the current configuration
     */
    const SocketConfig& config() const { return config_; }

private:
    DNNSocketInterface();
    ~DNNSocketInterface();

    mutable std::mutex mutex_;
    SocketConfig config_;
    std::unique_ptr<SocketClient> client_;
    std::unique_ptr<IPIProtocol> protocol_;
    bool hostInitialized_;
    std::vector<double> hostCell_;  // Cached cell from host init
    size_t calcCount_;
};

/**
 * @brief Parse socket configuration from string values
 *
 * @param typeStr Socket type ("unix" or "tcp")
 * @param address Socket address (path or host:port)
 * @return Parsed SocketConfig
 */
SocketConfig parseSocketConfig(const std::string& typeStr, const std::string& address);

} // namespace socket
} // namespace graspa

#endif // DNN_SOCKET_INTERFACE_H
