/**
 * @file IPIProtocol.cpp
 * @brief Implementation of modified iPI protocol for gRASPA-ASE communication
 */

#include "socket/IPIProtocol.h"
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <cmath>

namespace graspa {
namespace socket {

// Message header constants
static const std::string MSG_STATUS = "STATUS";
static const std::string MSG_READY = "READY";
static const std::string MSG_NEEDINIT = "NEEDINIT";
static const std::string MSG_HAVEDATA = "HAVEDATA";
static const std::string MSG_GETFORCE = "GETFORCE";
static const std::string MSG_FORCEREADY = "FORCEREADY";
static const std::string MSG_POSDATA = "POSDATA";
static const std::string MSG_HOSTDATA = "HOSTDATA";
static const std::string MSG_EXIT = "EXIT";

IPIProtocol::IPIProtocol(SocketClient& client)
    : client_(client)
    , state_(ProtocolState::Disconnected)
    , initialized_(false)
{
    if (client_.isConnected()) {
        state_ = ProtocolState::NeedInit;
    }
}

void IPIProtocol::initialize(const HostData& hostData) {
    if (!client_.isConnected()) {
        throw SocketError("Socket not connected");
    }

    // Wait for STATUS from server
    std::string header = client_.recvHeader();
    if (header != MSG_STATUS) {
        throw SocketError("Expected STATUS, got: " + header);
    }

    // Send NEEDINIT
    sendNeedInit();

    // Wait for server to request data (it doesn't, we just continue)
    // Send HOSTDATA
    sendHostData(hostData);

    // Send READY
    sendReady();

    state_ = ProtocolState::Ready;
    initialized_ = true;
}

double IPIProtocol::calculateEnergy(const PosData& posData) {
    if (!client_.isConnected()) {
        throw SocketError("Socket not connected");
    }

    // Wait for STATUS from server
    std::string header = client_.recvHeader();
    if (header.empty()) {
        throw SocketError("Connection closed by server");
    }
    if (header != MSG_STATUS) {
        throw SocketError("Expected STATUS, got: " + header);
    }

    // Send READY
    sendReady();

    // Wait for POSDATA request
    header = client_.recvHeader();
    if (header != MSG_POSDATA) {
        throw SocketError("Expected POSDATA request, got: " + header);
    }

    // Send position data
    sendPosData(posData);

    // Wait for HAVEDATA
    header = client_.recvHeader();
    if (header != MSG_HAVEDATA) {
        throw SocketError("Expected HAVEDATA, got: " + header);
    }

    // Send GETFORCE
    client_.sendHeader(MSG_GETFORCE);

    // Receive energy
    double energyHartree = recvForceReady();

    // Convert Hartree to kJ/mol
    double energyKJmol = energyHartree * HARTREE_TO_KJMOL;

    return energyKJmol;
}

void IPIProtocol::sendExit() {
    if (client_.isConnected()) {
        // Wait for STATUS if we're in a calculation cycle
        // Just send EXIT whenever we can
        client_.sendHeader(MSG_EXIT);
        state_ = ProtocolState::Disconnected;
    }
}

void IPIProtocol::sendNeedInit() {
    client_.sendHeader(MSG_NEEDINIT);
}

void IPIProtocol::sendReady() {
    client_.sendHeader(MSG_READY);
}

void IPIProtocol::sendHostData(const HostData& hostData) {
    // Send header
    client_.sendHeader(MSG_HOSTDATA);

    // Encode and send cell (9 doubles in Bohr)
    std::vector<double> cellBohr = encodeCellToBohr(hostData.cell);
    client_.sendAll(cellBohr.data(), cellBohr.size() * sizeof(double));

    // Send number of atoms
    int32_t natoms = hostData.numAtoms();
    client_.sendAll(&natoms, sizeof(natoms));

    if (natoms > 0) {
        // Encode and send positions (3*N doubles in Bohr)
        std::vector<double> posBohr = encodePositionsToBohr(hostData.positions);
        client_.sendAll(posBohr.data(), posBohr.size() * sizeof(double));

        // Send atom types (N int32)
        client_.sendAll(hostData.types.data(), hostData.types.size() * sizeof(int32_t));

        // Send charges (N doubles)
        client_.sendAll(hostData.charges.data(), hostData.charges.size() * sizeof(double));
    }

    // Send type map
    int32_t ntypes = static_cast<int32_t>(hostData.typeMap.size());
    client_.sendAll(&ntypes, sizeof(ntypes));

    for (const auto& kv : hostData.typeMap) {
        int32_t typeId = kv.first;
        client_.sendAll(&typeId, sizeof(typeId));

        int32_t symbolLen = static_cast<int32_t>(kv.second.length());
        client_.sendAll(&symbolLen, sizeof(symbolLen));

        client_.sendAll(kv.second.data(), symbolLen);
    }
}

void IPIProtocol::sendPosData(const PosData& posData) {
    // Send header
    client_.sendHeader(MSG_POSDATA);

    // Encode and send cell (9 doubles in Bohr)
    std::vector<double> cellBohr = encodeCellToBohr(posData.cell);
    client_.sendAll(cellBohr.data(), cellBohr.size() * sizeof(double));

    // Calculate and send inverse cell
    // For 3x3 matrix inversion
    std::vector<double> invCell(9);
    double det = cellBohr[0] * (cellBohr[4] * cellBohr[8] - cellBohr[5] * cellBohr[7])
               - cellBohr[1] * (cellBohr[3] * cellBohr[8] - cellBohr[5] * cellBohr[6])
               + cellBohr[2] * (cellBohr[3] * cellBohr[7] - cellBohr[4] * cellBohr[6]);

    if (std::abs(det) < 1e-10) {
        throw SocketError("Cell matrix is singular");
    }

    double invDet = 1.0 / det;
    invCell[0] = (cellBohr[4] * cellBohr[8] - cellBohr[5] * cellBohr[7]) * invDet;
    invCell[1] = (cellBohr[2] * cellBohr[7] - cellBohr[1] * cellBohr[8]) * invDet;
    invCell[2] = (cellBohr[1] * cellBohr[5] - cellBohr[2] * cellBohr[4]) * invDet;
    invCell[3] = (cellBohr[5] * cellBohr[6] - cellBohr[3] * cellBohr[8]) * invDet;
    invCell[4] = (cellBohr[0] * cellBohr[8] - cellBohr[2] * cellBohr[6]) * invDet;
    invCell[5] = (cellBohr[2] * cellBohr[3] - cellBohr[0] * cellBohr[5]) * invDet;
    invCell[6] = (cellBohr[3] * cellBohr[7] - cellBohr[4] * cellBohr[6]) * invDet;
    invCell[7] = (cellBohr[1] * cellBohr[6] - cellBohr[0] * cellBohr[7]) * invDet;
    invCell[8] = (cellBohr[0] * cellBohr[4] - cellBohr[1] * cellBohr[3]) * invDet;

    client_.sendAll(invCell.data(), invCell.size() * sizeof(double));

    // Send number of atoms
    int32_t natoms = posData.numAtoms();
    client_.sendAll(&natoms, sizeof(natoms));

    // Encode and send positions
    std::vector<double> posBohr = encodePositionsToBohr(posData.positions);
    client_.sendAll(posBohr.data(), posBohr.size() * sizeof(double));

    // Send atom types
    client_.sendAll(posData.types.data(), posData.types.size() * sizeof(int32_t));
}

double IPIProtocol::recvForceReady() {
    // Receive FORCEREADY header
    std::string header = client_.recvHeader();
    if (header != MSG_FORCEREADY) {
        throw SocketError("Expected FORCEREADY, got: " + header);
    }

    // Receive energy (1 double in Hartree)
    double energy;
    client_.recvAll(&energy, sizeof(energy));

    // Receive natoms (int32) - should be 0 (no forces)
    int32_t natoms;
    client_.recvAll(&natoms, sizeof(natoms));

    // Receive virial (9 doubles) - zeros
    std::vector<double> virial(9);
    client_.recvAll(virial.data(), virial.size() * sizeof(double));

    // Receive extra (int32) - should be 0
    int32_t extra;
    client_.recvAll(&extra, sizeof(extra));

    return energy;
}

std::vector<double> IPIProtocol::encodeCellToBohr(const std::vector<double>& cell) {
    if (cell.size() != 9) {
        throw std::invalid_argument("Cell must have 9 elements");
    }
    std::vector<double> result(9);
    for (size_t i = 0; i < 9; ++i) {
        result[i] = cell[i] * ANGSTROM_TO_BOHR;
    }
    return result;
}

std::vector<double> IPIProtocol::encodePositionsToBohr(const std::vector<double>& positions) {
    std::vector<double> result(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        result[i] = positions[i] * ANGSTROM_TO_BOHR;
    }
    return result;
}

} // namespace socket
} // namespace graspa
