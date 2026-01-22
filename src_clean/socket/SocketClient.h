/**
 * @file SocketClient.h
 * @brief Socket client for gRASPA-ASE communication
 *
 * Provides TCP and Unix domain socket connectivity for communicating
 * with the Python ASE calculator server.
 */

#ifndef SOCKET_CLIENT_H
#define SOCKET_CLIENT_H

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace graspa {
namespace socket {

/**
 * @brief Exception class for socket errors
 */
class SocketError : public std::runtime_error {
public:
    explicit SocketError(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * @brief Socket type enumeration
 */
enum class SocketType {
    Unix,   ///< Unix domain socket (local, faster)
    TCP     ///< TCP socket (network, HPC)
};

/**
 * @brief Socket client for communication with Python ASE server
 *
 * Supports both Unix domain sockets (for local, faster communication)
 * and TCP sockets (for HPC cross-node communication).
 */
class SocketClient {
public:
    /**
     * @brief Construct a new Socket Client
     */
    SocketClient();

    /**
     * @brief Destructor - closes connection if open
     */
    ~SocketClient();

    // Non-copyable
    SocketClient(const SocketClient&) = delete;
    SocketClient& operator=(const SocketClient&) = delete;

    // Movable
    SocketClient(SocketClient&& other) noexcept;
    SocketClient& operator=(SocketClient&& other) noexcept;

    /**
     * @brief Connect to a Unix domain socket
     * @param path Path to the Unix socket file
     * @throws SocketError on connection failure
     */
    void connectUnix(const std::string& path);

    /**
     * @brief Connect to a TCP socket
     * @param host Hostname or IP address
     * @param port Port number
     * @throws SocketError on connection failure
     */
    void connectTCP(const std::string& host, int port);

    /**
     * @brief Check if connected
     * @return true if connected
     */
    bool isConnected() const { return fd_ >= 0; }

    /**
     * @brief Close the connection
     */
    void close();

    /**
     * @brief Send all data
     * @param data Pointer to data buffer
     * @param size Number of bytes to send
     * @throws SocketError on failure
     */
    void sendAll(const void* data, size_t size);

    /**
     * @brief Send a string (as raw bytes)
     * @param str String to send
     * @throws SocketError on failure
     */
    void sendString(const std::string& str);

    /**
     * @brief Receive exactly size bytes
     * @param buffer Pointer to receive buffer
     * @param size Number of bytes to receive
     * @return Number of bytes received (0 if connection closed)
     * @throws SocketError on failure
     */
    size_t recvAll(void* buffer, size_t size);

    /**
     * @brief Receive a fixed-size string
     * @param size Number of bytes to receive
     * @return Received string
     * @throws SocketError on failure
     */
    std::string recvString(size_t size);

    /**
     * @brief Send a 12-byte header message
     * @param msg Header string (will be padded/truncated to 12 bytes)
     */
    void sendHeader(const std::string& msg);

    /**
     * @brief Receive a 12-byte header message
     * @return Header string (trimmed)
     */
    std::string recvHeader();

    /**
     * @brief Get the socket file descriptor
     * @return File descriptor or -1 if not connected
     */
    int fd() const { return fd_; }

    /**
     * @brief Get socket type
     * @return Current socket type
     */
    SocketType type() const { return type_; }

    /**
     * @brief Get socket address string
     * @return Address description (e.g., "unix:/tmp/sock" or "tcp:host:port")
     */
    const std::string& address() const { return address_; }

private:
    int fd_;                    ///< Socket file descriptor
    SocketType type_;           ///< Socket type
    std::string address_;       ///< Address string for logging

    static constexpr size_t HEADER_SIZE = 12;
};

} // namespace socket
} // namespace graspa

#endif // SOCKET_CLIENT_H
