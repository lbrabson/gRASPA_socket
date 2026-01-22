/**
 * @file IPIProtocol.h
 * @brief Modified iPI protocol implementation for gRASPA-ASE communication
 *
 * Implements the protocol state machine for communicating with the
 * Python ASE calculator server.
 */

#ifndef IPI_PROTOCOL_H
#define IPI_PROTOCOL_H

#include "socket/SocketClient.h"
#include <vector>
#include <map>
#include <string>
#include <cstdint>

namespace graspa {
namespace socket {

// Unit conversion constants (Angstrom <-> Bohr)
constexpr double BOHR_TO_ANGSTROM = 0.529177210903;
constexpr double ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM;

// Energy conversion constants
constexpr double HARTREE_TO_KJMOL = 2625.4996394799;
constexpr double KJMOL_TO_HARTREE = 1.0 / HARTREE_TO_KJMOL;

/**
 * @brief Host (framework) atom data
 */
struct HostData {
    std::vector<double> cell;       ///< 3x3 cell matrix (row-major, Angstrom)
    std::vector<double> positions;  ///< Nx3 positions (flattened, Angstrom)
    std::vector<int32_t> types;     ///< N atom type IDs
    std::vector<double> charges;    ///< N partial charges
    std::map<int32_t, std::string> typeMap;  ///< type_id -> symbol

    int numAtoms() const { return static_cast<int>(types.size()); }
};

/**
 * @brief Guest atom position data
 */
struct PosData {
    std::vector<double> cell;       ///< 3x3 cell matrix (row-major, Angstrom)
    std::vector<double> positions;  ///< Nx3 positions (flattened, Angstrom)
    std::vector<int32_t> types;     ///< N atom type IDs

    int numAtoms() const { return static_cast<int>(types.size()); }
};

/**
 * @brief Protocol state enumeration
 */
enum class ProtocolState {
    Disconnected,   ///< Not connected
    NeedInit,       ///< Need to send initialization
    Ready,          ///< Ready for calculations
    WaitingForData, ///< Waiting for position request
    HaveData,       ///< Data sent, waiting for energy
    Error           ///< Protocol error
};

/**
 * @brief iPI-style protocol handler for gRASPA-ASE communication
 *
 * Manages the communication state machine and handles message
 * encoding/decoding according to the modified iPI protocol.
 */
class IPIProtocol {
public:
    /**
     * @brief Construct protocol handler
     * @param client Reference to socket client (must remain valid)
     */
    explicit IPIProtocol(SocketClient& client);

    /**
     * @brief Get current protocol state
     */
    ProtocolState state() const { return state_; }

    /**
     * @brief Check if protocol is in a valid state for calculations
     */
    bool isReady() const { return state_ == ProtocolState::Ready; }

    /**
     * @brief Initialize connection with host data
     *
     * Sends NEEDINIT and HOSTDATA messages to the server.
     *
     * @param hostData Framework atom data
     * @throws SocketError on communication failure
     */
    void initialize(const HostData& hostData);

    /**
     * @brief Calculate energy for guest positions
     *
     * Handles the full request-response cycle:
     * 1. Wait for STATUS from server
     * 2. Send READY
     * 3. Wait for POSDATA request
     * 4. Send position data
     * 5. Wait for HAVEDATA
     * 6. Send GETFORCE
     * 7. Receive energy from FORCEREADY
     *
     * @param posData Guest atom positions
     * @return Energy in kJ/mol
     * @throws SocketError on communication failure
     */
    double calculateEnergy(const PosData& posData);

    /**
     * @brief Send exit message
     */
    void sendExit();

    /**
     * @brief Reset state after error
     */
    void reset() { state_ = ProtocolState::Disconnected; }

private:
    /**
     * @brief Send NEEDINIT message
     */
    void sendNeedInit();

    /**
     * @brief Send HOSTDATA message
     */
    void sendHostData(const HostData& hostData);

    /**
     * @brief Send READY message
     */
    void sendReady();

    /**
     * @brief Send POSDATA message
     */
    void sendPosData(const PosData& posData);

    /**
     * @brief Receive and parse FORCEREADY message
     * @return Energy in Hartree
     */
    double recvForceReady();

    /**
     * @brief Encode cell matrix to Bohr
     */
    std::vector<double> encodeCellToBohr(const std::vector<double>& cell);

    /**
     * @brief Encode positions to Bohr
     */
    std::vector<double> encodePositionsToBohr(const std::vector<double>& positions);

    SocketClient& client_;
    ProtocolState state_;
    bool initialized_;
};

} // namespace socket
} // namespace graspa

#endif // IPI_PROTOCOL_H
