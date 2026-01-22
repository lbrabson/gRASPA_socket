/**
 * @file socket_integration.cpp
 * @brief Implementation of bridge between gRASPA and socket interface
 */

#include "socket_integration.h"
#include <iostream>
#include <cstring>

// Internal helper: Copy positions from GPU Atoms struct to host vectors
static void CopyAtomsToVectors(Atoms& AtomData, size_t start, size_t count,
                                std::vector<double>& positions,
                                std::vector<int32_t>& types,
                                std::vector<double>& charges) {
    positions.resize(count * 3);
    types.resize(count);
    charges.resize(count);

    // Copy from GPU (Atoms struct has CUDA managed memory)
    for (size_t i = 0; i < count; ++i) {
        size_t idx = start + i;
        positions[i * 3 + 0] = AtomData.pos[idx].x;
        positions[i * 3 + 1] = AtomData.pos[idx].y;
        positions[i * 3 + 2] = AtomData.pos[idx].z;
        types[i] = static_cast<int32_t>(AtomData.Type[idx]);
        charges[i] = AtomData.charge[idx];
    }
}

std::vector<double> GetCellForSocket(Boxsize& Box) {
    std::vector<double> cell(9);
    // Box.Cell is a device pointer, need to copy to host
    double hostCell[9];
    cudaMemcpy(hostCell, Box.Cell, 9 * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 9; ++i) {
        cell[i] = hostCell[i];
    }
    return cell;
}

void PrepareGuestPositionsForSocket(Atoms& AtomData, size_t start, size_t size,
                                    std::vector<double>& positions,
                                    std::vector<int32_t>& types,
                                    Components& SystemComponents) {
    positions.resize(size * 3);
    types.resize(size);

    for (size_t i = 0; i < size; ++i) {
        size_t idx = start + i;
        positions[i * 3 + 0] = AtomData.pos[idx].x;
        positions[i * 3 + 1] = AtomData.pos[idx].y;
        positions[i * 3 + 2] = AtomData.pos[idx].z;
        types[i] = static_cast<int32_t>(AtomData.Type[idx]);
    }
}

void Socket_Initialize(Components& SystemComponents, Simulations& Sims, Boxsize& Box) {
    using namespace graspa::socket;

    auto& socketInterface = DNNSocketInterface::getInstance();

    // Parse configuration
    SocketConfig config = parseSocketConfig(SystemComponents.SocketType,
                                            SystemComponents.SocketAddress);
    socketInterface.configure(config);

    // Connect to server
    socketInterface.connect();

    // Extract framework (host) atom data
    // Framework is component 0
    Atoms& framework = SystemComponents.HostSystem[0];
    size_t nFrameworkAtoms = framework.size;

    std::vector<double> hostPositions(nFrameworkAtoms * 3);
    std::vector<int32_t> hostTypes(nFrameworkAtoms);
    std::vector<double> hostCharges(nFrameworkAtoms);

    for (size_t i = 0; i < nFrameworkAtoms; ++i) {
        hostPositions[i * 3 + 0] = framework.pos[i].x;
        hostPositions[i * 3 + 1] = framework.pos[i].y;
        hostPositions[i * 3 + 2] = framework.pos[i].z;
        hostTypes[i] = static_cast<int32_t>(framework.Type[i]);
        hostCharges[i] = framework.charge[i];
    }

    // Get cell matrix
    std::vector<double> cell = GetCellForSocket(Sims.Box);

    // Build type map from PseudoAtoms
    std::map<int32_t, std::string> typeMap;
    for (size_t i = 0; i < SystemComponents.PseudoAtoms.Symbol.size(); ++i) {
        typeMap[static_cast<int32_t>(i)] = SystemComponents.PseudoAtoms.Symbol[i];
    }

    // Initialize host on server
    socketInterface.initializeHost(cell, hostPositions, hostTypes, hostCharges, typeMap);

    std::cout << "[Socket] Initialized with " << nFrameworkAtoms << " framework atoms" << std::endl;
}

void Socket_Finalize() {
    using namespace graspa::socket;

    auto& socketInterface = DNNSocketInterface::getInstance();
    socketInterface.disconnect();
}

double Socket_Prediction_Total(Components& SystemComponents, Simulations& Sims) {
    using namespace graspa::socket;

    auto& socketInterface = DNNSocketInterface::getInstance();

    // Collect all guest atoms from all adsorbate components
    std::vector<double> guestPositions;
    std::vector<int32_t> guestTypes;

    // Start from component 1 (component 0 is framework)
    for (size_t comp = SystemComponents.NComponents.y; comp < static_cast<size_t>(SystemComponents.NComponents.x); ++comp) {
        Atoms& atoms = SystemComponents.HostSystem[comp];
        size_t nAtoms = atoms.size;

        for (size_t i = 0; i < nAtoms; ++i) {
            guestPositions.push_back(atoms.pos[i].x);
            guestPositions.push_back(atoms.pos[i].y);
            guestPositions.push_back(atoms.pos[i].z);
            guestTypes.push_back(static_cast<int32_t>(atoms.Type[i]));
        }
    }

    if (guestPositions.empty()) {
        return 0.0;  // No guests, zero energy
    }

    // Get cell matrix
    std::vector<double> cell = GetCellForSocket(Sims.Box);

    // Calculate energy (returns kJ/mol)
    double energy_kjmol = socketInterface.calculateGuestEnergy(guestPositions, guestTypes, cell);

    // Convert to internal units using the configured conversion factor
    double energy_internal = energy_kjmol * SystemComponents.DNNEnergyConversion;

    return energy_internal;
}

double Socket_Prediction_Move(Components& SystemComponents, Simulations& Sims,
                               size_t SelectedComponent, int MoveType) {
    using namespace graspa::socket;

    auto& socketInterface = DNNSocketInterface::getInstance();

    std::vector<double> positions;
    std::vector<int32_t> types;
    std::vector<double> cell = GetCellForSocket(Sims.Box);

    size_t molSize = SystemComponents.Moleculesize[SelectedComponent];

    switch (MoveType) {
        case INSERTION:
        case SINGLE_INSERTION: {
            // New configuration is in Sims.Old (after Prepare_DNN_InitialPositions)
            // For insertion, the new molecule positions start at index 0
            positions.resize(molSize * 3);
            types.resize(molSize);

            for (size_t i = 0; i < molSize; ++i) {
                positions[i * 3 + 0] = Sims.Old.pos[i].x;
                positions[i * 3 + 1] = Sims.Old.pos[i].y;
                positions[i * 3 + 2] = Sims.Old.pos[i].z;
                types[i] = static_cast<int32_t>(Sims.Old.Type[i]);
            }

            // For insertion, we compute absolute energy of new guest
            double energy_kjmol = socketInterface.calculateGuestEnergy(positions, types, cell);
            return energy_kjmol * SystemComponents.DNNEnergyConversion;
        }

        case DELETION:
        case SINGLE_DELETION: {
            // Old configuration (being deleted) is in Sims.Old
            positions.resize(molSize * 3);
            types.resize(molSize);

            for (size_t i = 0; i < molSize; ++i) {
                positions[i * 3 + 0] = Sims.Old.pos[i].x;
                positions[i * 3 + 1] = Sims.Old.pos[i].y;
                positions[i * 3 + 2] = Sims.Old.pos[i].z;
                types[i] = static_cast<int32_t>(Sims.Old.Type[i]);
            }

            // For deletion, we return negative of old energy (energy released)
            double energy_kjmol = socketInterface.calculateGuestEnergy(positions, types, cell);
            return -energy_kjmol * SystemComponents.DNNEnergyConversion;
        }

        case TRANSLATION:
        case ROTATION: {
            // For translation/rotation, Old has both old and new configs
            // Old positions: [0, molSize)
            // New positions: [molSize, 2*molSize)

            // Calculate old energy
            std::vector<double> oldPositions(molSize * 3);
            std::vector<int32_t> oldTypes(molSize);
            for (size_t i = 0; i < molSize; ++i) {
                oldPositions[i * 3 + 0] = Sims.Old.pos[i].x;
                oldPositions[i * 3 + 1] = Sims.Old.pos[i].y;
                oldPositions[i * 3 + 2] = Sims.Old.pos[i].z;
                oldTypes[i] = static_cast<int32_t>(Sims.Old.Type[i]);
            }
            double oldEnergy = socketInterface.calculateGuestEnergy(oldPositions, oldTypes, cell);

            // Calculate new energy
            std::vector<double> newPositions(molSize * 3);
            std::vector<int32_t> newTypes(molSize);
            for (size_t i = 0; i < molSize; ++i) {
                size_t idx = molSize + i;
                newPositions[i * 3 + 0] = Sims.Old.pos[idx].x;
                newPositions[i * 3 + 1] = Sims.Old.pos[idx].y;
                newPositions[i * 3 + 2] = Sims.Old.pos[idx].z;
                newTypes[i] = static_cast<int32_t>(Sims.Old.Type[idx]);
            }
            double newEnergy = socketInterface.calculateGuestEnergy(newPositions, newTypes, cell);

            // Return delta (new - old)
            return (newEnergy - oldEnergy) * SystemComponents.DNNEnergyConversion;
        }

        default:
            std::cerr << "[Socket] Unknown move type: " << MoveType << std::endl;
            return 0.0;
    }
}

double Socket_Prediction_Reinsertion(Components& SystemComponents, Simulations& Sims,
                                      size_t SelectedComponent, double3* temp) {
    using namespace graspa::socket;

    auto& socketInterface = DNNSocketInterface::getInstance();

    size_t molSize = SystemComponents.Moleculesize[SelectedComponent];
    std::vector<double> cell = GetCellForSocket(Sims.Box);

    // Old positions are at [0, molSize) in Sims.Old
    std::vector<double> oldPositions(molSize * 3);
    std::vector<int32_t> oldTypes(molSize);
    for (size_t i = 0; i < molSize; ++i) {
        oldPositions[i * 3 + 0] = Sims.Old.pos[i].x;
        oldPositions[i * 3 + 1] = Sims.Old.pos[i].y;
        oldPositions[i * 3 + 2] = Sims.Old.pos[i].z;
        oldTypes[i] = static_cast<int32_t>(Sims.Old.Type[i]);
    }
    double oldEnergy = socketInterface.calculateGuestEnergy(oldPositions, oldTypes, cell);

    // New positions are at [molSize, 2*molSize) in Sims.Old (after Prepare_DNN_InitialPositions_Reinsertion)
    std::vector<double> newPositions(molSize * 3);
    std::vector<int32_t> newTypes(molSize);
    for (size_t i = 0; i < molSize; ++i) {
        size_t idx = molSize + i;
        newPositions[i * 3 + 0] = Sims.Old.pos[idx].x;
        newPositions[i * 3 + 1] = Sims.Old.pos[idx].y;
        newPositions[i * 3 + 2] = Sims.Old.pos[idx].z;
        newTypes[i] = static_cast<int32_t>(Sims.Old.Type[idx]);
    }
    double newEnergy = socketInterface.calculateGuestEnergy(newPositions, newTypes, cell);

    // Return delta (new - old)
    return (newEnergy - oldEnergy) * SystemComponents.DNNEnergyConversion;
}
