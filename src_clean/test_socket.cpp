/**
 * @file test_socket.cpp
 * @brief Standalone test for gRASPA socket communication
 *
 * This test verifies the socket connection to the Python ASE server
 * without requiring the full gRASPA simulation infrastructure.
 *
 * Usage:
 *   1. Start the Python server:
 *      graspa-serve --calculator emt --unix /tmp/graspa.sock -v
 *
 *   2. Compile this test:
 *      g++ -O3 -std=c++17 -I. -c socket/SocketClient.cpp -o socket/SocketClient.o
 *      g++ -O3 -std=c++17 -I. -c socket/IPIProtocol.cpp -o socket/IPIProtocol.o
 *      g++ -O3 -std=c++17 -I. -c socket/DNNSocketInterface.cpp -o socket/DNNSocketInterface.o
 *      g++ -O3 -std=c++17 -I. test_socket.cpp socket/*.o -lpthread -o test_socket
 *
 *   3. Run the test:
 *      ./test_socket
 */

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include "socket/DNNSocketInterface.h"

void print_separator() {
    std::cout << "========================================" << std::endl;
}

int main(int argc, char* argv[]) {
    using namespace graspa::socket;

    // Default socket configuration
    std::string socketType = "unix";
    std::string socketAddress = "/tmp/graspa.sock";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tcp" && i + 1 < argc) {
            socketType = "tcp";
            socketAddress = argv[++i];
        } else if (arg == "--unix" && i + 1 < argc) {
            socketType = "unix";
            socketAddress = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --unix PATH    Use Unix socket (default: /tmp/graspa.sock)\n"
                      << "  --tcp HOST:PORT  Use TCP socket\n"
                      << "  --help         Show this help\n";
            return 0;
        }
    }

    print_separator();
    std::cout << "gRASPA Socket Integration Test" << std::endl;
    print_separator();

    try {
        auto& socket = DNNSocketInterface::getInstance();

        // Configure socket
        std::cout << "\n[1] Configuring socket: " << socketType << " -> " << socketAddress << std::endl;
        SocketConfig config = parseSocketConfig(socketType, socketAddress);
        socket.configure(config);

        // Connect to server
        std::cout << "\n[2] Connecting to Python server..." << std::endl;
        socket.connect();
        std::cout << "    Connected successfully!" << std::endl;

        // Initialize with test host data (simple cubic cell with 2 Cu atoms)
        std::cout << "\n[3] Initializing host (framework) data..." << std::endl;

        // 10 Angstrom cubic cell
        std::vector<double> cell = {
            10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0
        };

        // Two framework atoms at corners
        std::vector<double> hostPositions = {
            0.0, 0.0, 0.0,    // Atom 1
            5.0, 5.0, 5.0     // Atom 2
        };
        std::vector<int32_t> hostTypes = {0, 0};
        std::vector<double> hostCharges = {0.0, 0.0};

        // Type map: type 0 = Cu
        std::map<int32_t, std::string> typeMap;
        typeMap[0] = "Cu";

        socket.initializeHost(cell, hostPositions, hostTypes, hostCharges, typeMap);
        std::cout << "    Host initialized with " << hostTypes.size() << " atoms" << std::endl;

        // Test energy calculations
        std::cout << "\n[4] Testing energy calculations..." << std::endl;

        // Test 1: Single guest atom at center
        {
            std::vector<double> guestPos = {2.5, 2.5, 2.5};
            std::vector<int32_t> guestTypes = {0};

            double energy = socket.calculateGuestEnergy(guestPos, guestTypes, cell);
            std::cout << "    Test 1 - Single atom at (2.5, 2.5, 2.5):" << std::endl;
            std::cout << "             Energy = " << energy << " kJ/mol" << std::endl;
        }

        // Test 2: Two guest atoms
        {
            std::vector<double> guestPos = {
                2.5, 2.5, 2.5,
                7.5, 7.5, 7.5
            };
            std::vector<int32_t> guestTypes = {0, 0};

            double energy = socket.calculateGuestEnergy(guestPos, guestTypes, cell);
            std::cout << "    Test 2 - Two atoms:" << std::endl;
            std::cout << "             Energy = " << energy << " kJ/mol" << std::endl;
        }

        // Test 3: Move atom and check energy change
        {
            std::cout << "    Test 3 - Energy vs distance:" << std::endl;
            for (double d = 1.0; d <= 5.0; d += 1.0) {
                std::vector<double> guestPos = {d, d, d};
                std::vector<int32_t> guestTypes = {0};

                double energy = socket.calculateGuestEnergy(guestPos, guestTypes, cell);
                double dist = std::sqrt(3.0) * d;
                std::cout << "             d=" << d << " (r=" << dist << " A): E=" << energy << " kJ/mol" << std::endl;
            }
        }

        std::cout << "\n[5] Disconnecting..." << std::endl;
        socket.disconnect();

        print_separator();
        std::cout << "All tests passed!" << std::endl;
        print_separator();

    } catch (const SocketError& e) {
        std::cerr << "\nSocket Error: " << e.what() << std::endl;
        std::cerr << "\nMake sure the Python server is running:" << std::endl;
        std::cerr << "  graspa-serve --calculator emt --unix /tmp/graspa.sock -v" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
