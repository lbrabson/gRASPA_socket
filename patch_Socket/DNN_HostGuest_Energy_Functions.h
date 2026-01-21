#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

#include <execution>

#include "read_data.h"
//###PATCH_LCLIN_INCLUDE_HEADER###//

__global__ void Initialize_DNN_Positions(Atoms* d_a, Atoms New, Atoms Old, double3* temp, size_t Oldsize, size_t Newsize, size_t SelectedComponent, size_t Location, size_t chainsize, int MoveType, size_t CYCLE)
{
  size_t ij = blockIdx.x * blockDim.x + threadIdx.x;

  if(ij < (Newsize + Oldsize))
  {
    Initialize_Copy_Positions_Together(d_a, New, Old, temp, Oldsize, Newsize, SelectedComponent, Location, chainsize, MoveType);
  }
  /*
  //Zhao's note: need to think about changing this boolean to switch//
  if(MoveType == TRANSLATION || MoveType == ROTATION || MoveType == SINGLE_INSERTION || MoveType == SINGLE_DELETION) // Translation/Rotation/single_insertion/single_deletion //
  {
    //For Translation/Rotation, the Old positions are already in the Old struct, just need to put the New positions into Old, after the Old positions//
    for(size_t i = Oldsize; i < Oldsize + Newsize; i++) //chainsize here is the total size of the molecule for translation/rotation
    {
      Old.pos[i]           = New.pos[i - Oldsize];
      Old.scale[i]         = New.scale[i - Oldsize];
      Old.charge[i]        = New.charge[i - Oldsize];
      Old.scaleCoul[i]     = New.scaleCoul[i - Oldsize];
    }
  }
  else if(MoveType == INSERTION || MoveType == CBCF_INSERTION) // Insertion & Fractional Insertion //
  {
    //Put the trial orientations in New to Old, right after the first bead position//
    if (chainsize == 0)  //If single atom molecule, first bead position is still in New, move it to old//
    {
      Old.pos[0]       = New.pos[Location];
      Old.scale[0]     = New.scale[Location];
      Old.charge[0]    = New.charge[Location];
      Old.scaleCoul[0] = New.scaleCoul[Location];
    }
    for(size_t i = 0; i < chainsize; i++)
    {
      Old.pos[i + 1]       = New.pos[Location * chainsize + i];
      Old.scale[i + 1]     = New.scale[Location * chainsize + i];
      Old.charge[i + 1]    = New.charge[Location * chainsize + i];
      Old.scaleCoul[i + 1] = New.scaleCoul[Location * chainsize + i];
    }
  }
  else if(MoveType == DELETION || MoveType == CBCF_DELETION) // Deletion //
  {
    for(size_t i = 0; i < Oldsize; i++)
    {
      // For deletion, Location = UpdateLocation, see Deletion Move //
      Old.pos[i]           = d_a[SelectedComponent].pos[Location + i];
      Old.scale[i]         = d_a[SelectedComponent].scale[Location + i];
      Old.charge[i]        = d_a[SelectedComponent].charge[Location + i];
      Old.scaleCoul[i]     = d_a[SelectedComponent].scaleCoul[Location + i];
    }
  }
  */

  /*
  if(CYCLE == 145) 
  {
  for(size_t i = 0; i < Oldsize + Newsize; i++)
  printf("Old pos: %.5f %.5f %.5f, scale/charge/scaleCoul: %.5f %.5f %.5f\n", Old.pos[i].x, Old.pos[i].y, Old.pos[i].z, Old.scale[i], Old.charge[i], Old.scaleCoul[i]);
  }
  */
}

void Prepare_DNN_InitialPositions(Atoms*& d_a, Atoms& New, Atoms& Old, double3* temp, Components& SystemComponents, size_t SelectedComponent, int MoveType, size_t Location)
{
  size_t Oldsize = 0; size_t Newsize = 0; size_t chainsize = 0;
  switch(MoveType)
  {
    case TRANSLATION: case ROTATION: // Translation/Rotation Move //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      chainsize = SystemComponents.Moleculesize[SelectedComponent];
      break;
    }
    case INSERTION: case SINGLE_INSERTION: // Insertion //
    {
      Oldsize   = 0;
      Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
    case DELETION:  case SINGLE_DELETION: // Deletion //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = 0;
      chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
    case REINSERTION: // Reinsertion //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      chainsize = SystemComponents.Moleculesize[SelectedComponent];
      //throw std::runtime_error("Use the Special Function for Reinsertion");
      break;
    }
    case IDENTITY_SWAP:
    {
      throw std::runtime_error("Use the Special Function for IDENTITY SWAP!");
    }
    case CBCF_LAMBDACHANGE: // CBCF Lambda Change //
    {
      throw std::runtime_error("Use the Special Function for CBCF Lambda Change");
      //Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      //Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      //chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      //break;
    }
    case CBCF_INSERTION: // CBCF Lambda Insertion //
    {
      Oldsize      = 0;
      Newsize      = SystemComponents.Moleculesize[SelectedComponent];
      chainsize    = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
    case CBCF_DELETION: // CBCF Lambda Deletion //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = 0;
      chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
  }
  //Initialize_DNN_Positions<<<1,1>>>(d_a, New, Old, Oldsize, Newsize, SelectedComponent, Location, chainsize, MoveType, SystemComponents.CURRENTCYCLE);
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock(Oldsize + Newsize, Nblock, Nthread);
  Initialize_DNN_Positions<<<Nblock,Nthread>>>(d_a, New, Old, SystemComponents.tempMolStorage, Oldsize, Newsize, SelectedComponent, Location, chainsize, MoveType, SystemComponents.CURRENTCYCLE);
}

__global__ void Initialize_DNN_Positions_Reinsertion(double3* temp, Atoms* d_a, Atoms Old, size_t Oldsize, size_t Newsize, size_t realpos, size_t SelectedComponent)
{
  for(size_t i = 0; i < Oldsize; i++)
  {
    Old.pos[i]       = d_a[SelectedComponent].pos[realpos + i];
    Old.scale[i]     = d_a[SelectedComponent].scale[realpos + i];
    Old.charge[i]    = d_a[SelectedComponent].charge[realpos + i];
    Old.scaleCoul[i] = d_a[SelectedComponent].scaleCoul[realpos + i];
  }
  //Reinsertion New Positions stored in three arrays, other data are the same as the Old molecule information in d_a//
  for(size_t i = Oldsize; i < Oldsize + Newsize; i++) //chainsize here is the total size of the molecule for translation/rotation
  {
    Old.pos[i]       = temp[i - Oldsize];
    Old.scale[i]     = d_a[SelectedComponent].scale[realpos + i - Oldsize];
    Old.charge[i]    = d_a[SelectedComponent].charge[realpos + i - Oldsize];
    Old.scaleCoul[i] = d_a[SelectedComponent].scaleCoul[realpos + i - Oldsize];
  }
}

void Prepare_DNN_InitialPositions_Reinsertion(Atoms*& d_a, Atoms& Old, double3* temp, Components& SystemComponents, size_t SelectedComponent, size_t Location)
{
  size_t numberOfAtoms = SystemComponents.Moleculesize[SelectedComponent];
  size_t Oldsize = 0; size_t Newsize = numberOfAtoms;
  //Zhao's note: translation/rotation/reinsertion involves new + old states. Insertion/Deletion only has the new state.
  Oldsize         = SystemComponents.Moleculesize[SelectedComponent];
  numberOfAtoms  += Oldsize;
  Initialize_DNN_Positions_Reinsertion<<<1,1>>>(temp, d_a, Old, Oldsize, Newsize, Location, SelectedComponent);
}
//###PATCH_ALLEGRO_CONSIDER_DNN_ATOMS###//
//###PATCH_SOCKET_CONSIDER_DNN_ATOMS_PATCHED###//
void Check_DNNAtom_and_copy_pos_to_UCAtoms(double3* temp_pos, Atoms& UCAtoms, bool* ConsiderThisAdsorbateAtom, size_t Molsize)
{
  size_t update_i = 0;
  for(size_t i = 0; i < Molsize; i++)
  {
    if(ConsiderThisAdsorbateAtom[i])
      UCAtoms.pos[update_i] = temp_pos[i];
    update_i ++;
  }
}


double DNN_Prediction_Move(Components& SystemComponents, Simulations& Sims, size_t SelectedComponent, int MoveType)
{
  switch(MoveType)
  {
  case INSERTION:
  {
    double DNN_New = 0.0;
    //###PATCH_ALLEGRO_INSERTION###//
    //###PATCH_LCLIN_INSERTION###//
    //###PATCH_SOCKET_INSERTION_PATCHED###//
    if(SystemComponents.UseSocket)
    {
      bool Initialize = false;
      double3* temp_pos; temp_pos = (double3*) malloc(sizeof(double3) * SystemComponents.Moleculesize[SelectedComponent]);
      cudaMemcpy(temp_pos, Sims.Old.pos, sizeof(double3) * SystemComponents.Moleculesize[SelectedComponent], cudaMemcpyDeviceToHost);
      Check_DNNAtom_and_copy_pos_to_UCAtoms(temp_pos, SystemComponents.DNN.UCAtoms[SelectedComponent], SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.Moleculesize[SelectedComponent]);
//      if(SystemComponents.CURRENTCYCLE <= 30)
//      {
//        printf("DNN INSERTION BEFORE CHECK DNN PseudoAtoms\n");
//        for(size_t i = 0; i < SystemComponents.Moleculesize[SelectedComponent]; i++)
//          printf("pos: %f %f %f\n", temp_pos[i].x, temp_pos[i].y, temp_pos[i].z);
//
//        printf("DNN INSERTION AFTER CHECK DNN PseudoAtoms\n");
//        for(size_t i = 0; i < SystemComponents.DNN.UCAtoms[SelectedComponent].size; i++)
//          printf("pos: %f %f %f\n", SystemComponents.DNN.UCAtoms[SelectedComponent].pos[i].x, SystemComponents.DNN.UCAtoms[SelectedComponent].pos[i].y, SystemComponents.DNN.UCAtoms[SelectedComponent].pos[i].z);
//      }
      DNN_New = SystemComponents.DNN.MCEnergyWrapper(SelectedComponent, Initialize, SystemComponents.DNNEnergyConversion);
      free(temp_pos);
    }

    return DNN_New;
  }
  case DELETION:
  {
    double DNN_New = 0.0;
    //###PATCH_ALLEGRO_DELETION###//
    //###PATCH_LCLIN_DELETION###//
    //###PATCH_SOCKET_DELETION_PATCHED###//
    if(SystemComponents.UseSocket)
    {
      bool Initialize = false;
      double3* temp_pos; temp_pos = (double3*) malloc(sizeof(double3) * SystemComponents.Moleculesize[SelectedComponent]);
      cudaMemcpy(temp_pos, Sims.Old.pos, sizeof(double3) * SystemComponents.Moleculesize[SelectedComponent], cudaMemcpyDeviceToHost);
      Check_DNNAtom_and_copy_pos_to_UCAtoms(temp_pos, SystemComponents.DNN.UCAtoms[SelectedComponent], SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.Moleculesize[SelectedComponent]);
      DNN_New = SystemComponents.DNN.MCEnergyWrapper(SelectedComponent, Initialize, SystemComponents.DNNEnergyConversion);
      free(temp_pos);
    }

    return DNN_New;
  }
  case TRANSLATION: case ROTATION: case SINGLE_INSERTION: case SINGLE_DELETION:
  {
    double DNN_New = 0.0; double DNN_Old = 0.0;
    //###PATCH_ALLEGRO_SINGLE###//
    //###PATCH_LCLIN_SINGLE###//
    //###PATCH_SOCKET_SINGLE_PATCHED###//
    bool Do_New = true; bool Do_Old = true;
    if(MoveType == SINGLE_INSERTION) Do_Old = false;
    if(MoveType == SINGLE_DELETION)  Do_New = false;
    if(SystemComponents.UseSocket)
    {
      bool Initialize = false;
      double3* temp_pos; temp_pos = (double3*) malloc(sizeof(double3) * SystemComponents.Moleculesize[SelectedComponent]);
      if(Do_New)
      {
        cudaMemcpy(temp_pos, Sims.New.pos, sizeof(double3) * SystemComponents.Moleculesize[SelectedComponent], cudaMemcpyDeviceToHost);
        Check_DNNAtom_and_copy_pos_to_UCAtoms(temp_pos, SystemComponents.DNN.UCAtoms[SelectedComponent], SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.Moleculesize[SelectedComponent]);
        DNN_New = SystemComponents.DNN.MCEnergyWrapper(SelectedComponent, Initialize, SystemComponents.DNNEnergyConversion);
        //printf("DNN_New %f\n", DNN_New);
      }
      if(Do_Old)
      {
        cudaMemcpy(temp_pos, Sims.Old.pos, sizeof(double3) * SystemComponents.Moleculesize[SelectedComponent], cudaMemcpyDeviceToHost);
        Check_DNNAtom_and_copy_pos_to_UCAtoms(temp_pos, SystemComponents.DNN.UCAtoms[SelectedComponent], SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.Moleculesize[SelectedComponent]);
        DNN_Old = SystemComponents.DNN.MCEnergyWrapper(SelectedComponent, Initialize, SystemComponents.DNNEnergyConversion);
        //printf("DNN_New %f\n", DNN_Old);
      }
      free(temp_pos);
    }

    return DNN_New - DNN_Old;
  }
  }
  return 0.0;
}

double DNN_Prediction_Reinsertion(Components& SystemComponents, Simulations& Sims, size_t SelectedComponent, double3* temp)
{
  double DNN_New = 0.0; double DNN_Old = 0.0;
  //###PATCH_ALLEGRO_REINSERTION###//
  //###PATCH_LCLIN_REINSERTION###//
  //###PATCH_SOCKET_REINSERTION_PATCHED###//
  if(SystemComponents.UseSocket)
  {
    bool Initialize = false;
    double3* temp_pos; temp_pos = (double3*) malloc(sizeof(double3) * SystemComponents.Moleculesize[SelectedComponent]);
    //NEW//
    cudaMemcpy(temp_pos, temp, sizeof(double3) * SystemComponents.Moleculesize[SelectedComponent], cudaMemcpyDeviceToHost);
    Check_DNNAtom_and_copy_pos_to_UCAtoms(temp_pos, SystemComponents.DNN.UCAtoms[SelectedComponent], SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.Moleculesize[SelectedComponent]);
    DNN_New = SystemComponents.DNN.MCEnergyWrapper(SelectedComponent, Initialize, SystemComponents.DNNEnergyConversion);
    //OLD//
    cudaMemcpy(temp_pos, Sims.Old.pos, sizeof(double3) * SystemComponents.Moleculesize[SelectedComponent], cudaMemcpyDeviceToHost);
    Check_DNNAtom_and_copy_pos_to_UCAtoms(temp_pos, SystemComponents.DNN.UCAtoms[SelectedComponent], SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.Moleculesize[SelectedComponent]);
    DNN_Old = SystemComponents.DNN.MCEnergyWrapper(SelectedComponent, Initialize, SystemComponents.DNNEnergyConversion);
    free(temp_pos);
  }

  return DNN_New - DNN_Old;
}

double DNN_Prediction_Total(Components& SystemComponents, Simulations& Sims)
{
  double DNN_E = 0.0;
  //###PATCH_ALLEGRO_FXNMAIN###//
  //###PATCH_LCLIN_FXNMAIN###//
  //###PATCH_SOCKET_FXNMAIN_PATCHED###//
  if(SystemComponents.UseSocket)
  {
    bool Initialize = false;
    size_t comp = 1;
    for(size_t i = 0; i < SystemComponents.NumberOfMolecule_for_Component[comp]; i++)
    {
      size_t update_i = 0;
      for(size_t j = 0; j < SystemComponents.Moleculesize[comp]; j++)
      {
        if(!SystemComponents.ConsiderThisAdsorbateAtom[j]) continue;
        size_t AtomIdx = i * SystemComponents.Moleculesize[comp] + j;
        SystemComponents.DNN.UCAtoms[comp].pos[update_i] = SystemComponents.HostSystem[comp].pos[AtomIdx];
        update_i ++;
      }
      DNN_E += SystemComponents.DNN.MCEnergyWrapper(comp, Initialize, SystemComponents.DNNEnergyConversion);
    }
  }
  return DNN_E;
}

void WriteOutliers(Components& SystemComponents, Simulations& Sim, int MoveType, MoveEnergy E, double Correction)
{
  //Write to a file for checking//
  std::ofstream textrestartFile{};
  std::string dirname="DNN/";
  std::string TRname  = dirname + "/" + "Outliers_SINGLE_PARTICLE.data";
  std::string Ifname   = dirname + "/" + "Outliers_INSERTION.data";
  std::string Dfname   = dirname + "/" + "Outliers_DELETION.data";


  std::filesystem::path cwd = std::filesystem::current_path();

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path IfileName   = cwd /Ifname;
  std::filesystem::path DfileName = cwd /Dfname;
  std::filesystem::path TRfileName = cwd /TRname;
  std::filesystem::create_directories(directoryName);

  size_t size = SystemComponents.HostSystem[1].Molsize;
  size_t ads_comp = 1;
  size_t start= 0;
  std::string Move;
  switch(MoveType)
  {
  case OLD: //Positions are stored in Sim.Old
  {
    SystemComponents.Copy_GPU_Data_To_Temp(Sim.Old, start, size);
    Move = "TRANSLATION_ROTATION_NEW_NON_CBMC_INSERTION";
    textrestartFile = std::ofstream(TRfileName, std::ios::app);
    break;
  }
  case NEW:
  {
    SystemComponents.Copy_GPU_Data_To_Temp(Sim.New, start, size);
    Move = "TRANSLATION_ROTATION_OLD_NON_CBMC_DELETION";
    textrestartFile = std::ofstream(TRfileName, std::ios::app);
    break;
  }
  case REINSERTION_OLD:
  {
    SystemComponents.Copy_GPU_Data_To_Temp(Sim.Old, start, size);
    Move = "REINSERTION_OLD";
    textrestartFile = std::ofstream(TRfileName, std::ios::app);
    break;
  }
  case REINSERTION_NEW:
  {
    start = SystemComponents.Moleculesize[ads_comp];
    SystemComponents.Copy_GPU_Data_To_Temp(Sim.Old, start, size);
    Move = "REINSERTION_NEW";
    textrestartFile = std::ofstream(TRfileName, std::ios::app);
    break;
  }
  case DNN_INSERTION:
  {
    SystemComponents.Copy_GPU_Data_To_Temp(Sim.Old, start, size);
    Move = "SWAP_INSERTION";
    textrestartFile = std::ofstream(IfileName, std::ios::app);
    break;
  }
  case DNN_DELETION:
  {
    SystemComponents.Copy_GPU_Data_To_Temp(Sim.Old, start, size);
    Move = "SWAP_DELETION";
    textrestartFile = std::ofstream(DfileName, std::ios::app);
    break;
  }
  }
  for(size_t i = 0; i < size; i++)
    textrestartFile << SystemComponents.TempSystem.pos[i].x << " " << SystemComponents.TempSystem.pos[i].y << " " << SystemComponents.TempSystem.pos[i].z << " " << SystemComponents.TempSystem.Type[i] << " " << Move << " " << E.DNN_E << " " << Correction << '\n';
  textrestartFile.close();
}

bool Check_DNN_Drift(Variables& Vars, size_t systemId, MoveEnergy& tot)
{ 
  Components& SystemComponents = Vars.SystemComponents[systemId];
  Simulations& Sims            = Vars.Sims[systemId];
  int&    MoveType             = SystemComponents.TempVal.MoveType;

  bool REJECT = false; 
  double correction = tot.DNN_Correction(); //If use DNN, HGVDWReal and HGEwaldE are zeroed//
  if(fabs(correction) > SystemComponents.DNNDrift) //If there is a huge drift in the energy correction between DNN and Classical HostGuest//
  { 
    //printf("TRANSLATION/ROTATION: Bad Prediction, reject the move!!!\n");
    switch(MoveType)
    {
      case TRANSLATION: case ROTATION:
      {
        SystemComponents.TranslationRotationDNNReject ++; break;
      }
      case SINGLE_INSERTION: case SINGLE_DELETION:
      {
        SystemComponents.SingleSwapDNNReject ++; break;
      }
    }
    WriteOutliers(SystemComponents, Sims, NEW, tot, correction); //Print New Locations//
    WriteOutliers(SystemComponents, Sims, OLD, tot, correction); //Print Old Locations//
    REJECT = true;
  } 
  return REJECT;
}
