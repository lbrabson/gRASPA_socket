/**
 * @file SocketClient.cpp
 * @brief Implementation of socket client for gRASPA-ASE communication
 */

#include "socket/SocketClient.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>
#include <sstream>

namespace graspa {
namespace socket {

SocketClient::SocketClient()
    : fd_(-1)
    , type_(SocketType::Unix)
    , address_()
{
}

SocketClient::~SocketClient() {
    close();
}

SocketClient::SocketClient(SocketClient&& other) noexcept
    : fd_(other.fd_)
    , type_(other.type_)
    , address_(std::move(other.address_))
{
    other.fd_ = -1;
}

SocketClient& SocketClient::operator=(SocketClient&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        type_ = other.type_;
        address_ = std::move(other.address_);
        other.fd_ = -1;
    }
    return *this;
}

void SocketClient::connectUnix(const std::string& path) {
    close();

    // Create Unix socket
    fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd_ < 0) {
        throw SocketError("Failed to create Unix socket: " + std::string(strerror(errno)));
    }

    // Setup address
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;

    if (path.length() >= sizeof(addr.sun_path)) {
        ::close(fd_);
        fd_ = -1;
        throw SocketError("Unix socket path too long: " + path);
    }
    strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    // Connect
    if (::connect(fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        int err = errno;
        ::close(fd_);
        fd_ = -1;
        throw SocketError("Failed to connect to Unix socket " + path + ": " + strerror(err));
    }

    type_ = SocketType::Unix;
    address_ = "unix:" + path;
}

void SocketClient::connectTCP(const std::string& host, int port) {
    close();

    // Create TCP socket
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        throw SocketError("Failed to create TCP socket: " + std::string(strerror(errno)));
    }

    // Resolve hostname
    struct hostent* server = gethostbyname(host.c_str());
    if (!server) {
        ::close(fd_);
        fd_ = -1;
        throw SocketError("Failed to resolve hostname: " + host);
    }

    // Setup address
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    addr.sin_port = htons(port);

    // Connect
    if (::connect(fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        int err = errno;
        ::close(fd_);
        fd_ = -1;
        std::ostringstream oss;
        oss << "Failed to connect to " << host << ":" << port << ": " << strerror(err);
        throw SocketError(oss.str());
    }

    type_ = SocketType::TCP;
    std::ostringstream oss;
    oss << "tcp:" << host << ":" << port;
    address_ = oss.str();
}

void SocketClient::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    address_.clear();
}

void SocketClient::sendAll(const void* data, size_t size) {
    if (fd_ < 0) {
        throw SocketError("Socket not connected");
    }

    const char* ptr = static_cast<const char*>(data);
    size_t remaining = size;

    while (remaining > 0) {
        ssize_t sent = ::send(fd_, ptr, remaining, 0);
        if (sent < 0) {
            if (errno == EINTR) continue;
            throw SocketError("Send failed: " + std::string(strerror(errno)));
        }
        if (sent == 0) {
            throw SocketError("Connection closed during send");
        }
        ptr += sent;
        remaining -= sent;
    }
}

void SocketClient::sendString(const std::string& str) {
    sendAll(str.data(), str.size());
}

size_t SocketClient::recvAll(void* buffer, size_t size) {
    if (fd_ < 0) {
        throw SocketError("Socket not connected");
    }

    char* ptr = static_cast<char*>(buffer);
    size_t remaining = size;
    size_t total = 0;

    while (remaining > 0) {
        ssize_t received = ::recv(fd_, ptr, remaining, 0);
        if (received < 0) {
            if (errno == EINTR) continue;
            throw SocketError("Receive failed: " + std::string(strerror(errno)));
        }
        if (received == 0) {
            // Connection closed
            if (total == 0) {
                return 0;
            }
            throw SocketError("Connection closed during receive");
        }
        ptr += received;
        remaining -= received;
        total += received;
    }

    return total;
}

std::string SocketClient::recvString(size_t size) {
    std::string result(size, '\0');
    size_t received = recvAll(&result[0], size);
    if (received == 0) {
        return "";
    }
    return result;
}

void SocketClient::sendHeader(const std::string& msg) {
    std::string header = msg;
    // Pad to 12 bytes with spaces
    header.resize(HEADER_SIZE, ' ');
    sendAll(header.data(), HEADER_SIZE);
}

std::string SocketClient::recvHeader() {
    std::string header = recvString(HEADER_SIZE);
    if (header.empty()) {
        return "";
    }
    // Trim trailing spaces
    size_t end = header.find_last_not_of(' ');
    if (end != std::string::npos) {
        header = header.substr(0, end + 1);
    } else {
        header.clear();
    }
    return header;
}

} // namespace socket
} // namespace graspa
