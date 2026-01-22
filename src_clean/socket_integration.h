/**
 * @file socket_integration.h
 * @brief Bridge between gRASPA data structures and socket interface
 *
 * Provides functions to convert gRASPA's GPU-based Atoms struct to the
 * socket interface's std::vector format for communication with the
 * Python ASE calculator server.
 */

#ifndef SOCKET_INTEGRATION_H
#define SOCKET_INTEGRATION_H

#include "socket/DNNSocketInterface.h"
#include "data_struct.h"

/**
 * @brief Initialize socket connection and send host (framework) data
 *
 * Must be called once after framework is loaded and before any MC moves.
 * Extracts framework atom data from GPU and sends to Python server.
 *
 * @param SystemComponents Component data containing framework info
 * @param Sims Simulation data containing atom positions
 * @param Box Simulation box parameters
 */
void Socket_Initialize(Components& SystemComponents, Simulations& Sims, Boxsize& Box);

/**
 * @brief Finalize socket connection
 *
 * Sends exit message and closes connection to Python server.
 * Should be called at end of simulation.
 */
void Socket_Finalize();

/**
 * @brief Calculate total DNN energy via socket
 *
 * Computes total energy for all guest molecules in the system.
 *
 * @param SystemComponents Component data
 * @param Sims Simulation data
 * @return Total DNN energy in internal units (converted from kJ/mol)
 */
double Socket_Prediction_Total(Components& SystemComponents, Simulations& Sims);

/**
 * @brief Calculate DNN energy for MC move
 *
 * Computes energy for insertion, deletion, translation, or rotation moves.
 * Uses data prepared in Sims.Old and Sims.New structures.
 *
 * @param SystemComponents Component data
 * @param Sims Simulation data
 * @param SelectedComponent Component index being modified
 * @param MoveType Type of MC move (INSERTION, DELETION, TRANSLATION, etc.)
 * @return Energy in internal units (delta for single moves, absolute for swap)
 */
double Socket_Prediction_Move(Components& SystemComponents, Simulations& Sims,
                               size_t SelectedComponent, int MoveType);

/**
 * @brief Calculate DNN energy for reinsertion move
 *
 * Computes energy change for molecule reinsertion using temp positions.
 *
 * @param SystemComponents Component data
 * @param Sims Simulation data
 * @param SelectedComponent Component index being reinserted
 * @param temp Temporary positions for new configuration
 * @return Energy difference (new - old) in internal units
 */
double Socket_Prediction_Reinsertion(Components& SystemComponents, Simulations& Sims,
                                      size_t SelectedComponent, double3* temp);

/**
 * @brief Extract guest positions from GPU to socket format
 *
 * Helper function to copy guest atom data from GPU Atoms struct
 * to vectors suitable for socket communication.
 *
 * @param Atoms GPU atom data structure
 * @param start Starting index in Atoms array
 * @param size Number of atoms to extract
 * @param positions Output: flattened Nx3 positions (Angstrom)
 * @param types Output: atom type IDs
 * @param SystemComponents Component data for type mapping
 */
void PrepareGuestPositionsForSocket(Atoms& AtomData, size_t start, size_t size,
                                    std::vector<double>& positions,
                                    std::vector<int32_t>& types,
                                    Components& SystemComponents);

/**
 * @brief Get cell matrix from Box in socket format
 *
 * Converts box Cell array to flattened 9-element vector.
 *
 * @param Box Simulation box
 * @return 3x3 cell matrix (row-major, Angstrom)
 */
std::vector<double> GetCellForSocket(Boxsize& Box);

#endif // SOCKET_INTEGRATION_H
