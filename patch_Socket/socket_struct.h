#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <errno.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

struct Socket
{
    Boxsize UCBox;
    Boxsize ReplicaBox;
    std::vector<Atoms> UCAtoms;
    std::vector<Atoms> ReplicaAtoms;
    std::vector<std::string> ElementSymbolUsed;
    std::vector<int>Match_Element_PseudoAtom_order; //length = # of PseudoAtoms, value stored = order in the Socket//

    int3 NReplicacell = {1,1,1};

    char socket_path[108] = "/tmp/ase_ipi_socket";  // UNIX path limit
    int    fd     = -1;   // UNIX socket file descriptor
    size_t natoms = 0;    // number of atoms sent in last call

    size_t DNN_Molsize = 0; //Atom size to be considered for DNN, since there might be fictional atom sites for a classical sim molecule

    size_t nstep = 0;
  
    /* Constructor-style init */
    void init(const char *path, int n_atoms)
    {
        fd = -1;
        natoms = n_atoms;
        strncpy(socket_path, path, sizeof(socket_path) - 1);
        socket_path[sizeof(socket_path) - 1] = '\0';
    }

    /* Connect to i-PI */
    int connect_socket()
    {
        struct sockaddr_un addr;

        fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) {
            perror("socket");
            return -1;
        }

        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            perror("connect");
            close(fd);
            fd = -1;
            return -1;
        }

        return 0;
    }

    /* Robust write */
    ssize_t write_all(const void *buf, size_t count)
    {
        size_t left = count;
        const char *ptr = (const char *)buf;

        while (left > 0) {
            ssize_t w = write(fd, ptr, left);
            if (w <= 0) {
                perror("write");
                return -1;
            }
            left -= w;
            ptr  += w;
        }
        return count;
    }

    /* Robust read */
ssize_t read_all(void *buf, size_t count)
{
    size_t left = count;
    char *ptr = (char *)buf;

    while (left > 0) {
        ssize_t r = read(fd, ptr, left);

        if (r == 0) {
            fprintf(stderr, "SOCKET CLOSED BY ASE — BAD PACKET SENT\n");
            exit(EXIT_FAILURE);
        }

        if (r < 0) {
            perror("read error");
            exit(EXIT_FAILURE);
        }

        left -= r;
        ptr  += r;
    }
    return count;
}


    /* Send ASCII command */
    int send_command(const char *cmd)
    {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "%s\n", cmd);
        return write_all(buffer, strlen(buffer));
    }

    /* Send positions */
    int send_positions(const double *xyz)
    {
        send_command("POSDATA");

        int n = (int)natoms;
        write_all(&n, sizeof(int));

        // send cell (ASE requires this)
        write_all(ReplicaBox.Cell, sizeof(double) * 9);

        write_all(xyz, sizeof(double) * 3 * natoms);

        return 0;
    }

    /* Receive energy */
    // int receive_energy(double *energy_ev)
    // {
    //     char header[8];

    //     // Read header ("FORCES\n") as i-PI sends it
    //     if (read_all(header, 7) != 7) {
    //         perror("read header");
    //         return -1;
    //     }
    //     header[7] = '\0';

    //     if (strncmp(header, "FORCES", 6) != 0) {
    //         fprintf(stderr, "Unexpected i-PI response: %s\n", header);
    //         return -1;
    //     }

    //     // Read only the energy (double)
    //     if (read_all(energy_ev, sizeof(double)) != sizeof(double)) {
    //         perror("read energy");
    //         return -1;
    //     }

    //     return 0;
    // }

    int receive_energy(double *energy_ev)
    {
        // 1. Energy
        if (read_all(energy_ev, sizeof(double)) != sizeof(double)) {
            perror("read energy");
            return -1;
        }

        // 2. Forces (discard if unused)
        std::vector<double> forces(3 * natoms);
        if (read_all(forces.data(), sizeof(double) * 3 * natoms) <= 0) {
            perror("read forces");
            return -1;
        }

        // 3. Stress (6 doubles)
        double stress[6];
        if (read_all(stress, sizeof(double) * 6) <= 0) {
            perror("read stress");
            return -1;
        }

        return 0;
    }

    /* Send coords and receive single energy */
    double PredictFromSocket(const double* xyz, size_t n_atoms)
    {
        natoms = n_atoms;   // update current atom count for socket

        // debugging by printing positions
        printf("[CLIENT] Sending POSDATA, natoms = %zu\n", natoms);
        for (size_t i = 0; i < std::min(natoms, (size_t)5); i++) {
            printf("  atom %zu: %f %f %f\n",
                i, xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
        }
        fflush(stderr);

        // send positions
        send_positions(xyz);

        double energy_ev;
        receive_energy(&energy_ev); // reads just energy from i-PI socket

        return energy_ev;
    }

    /* Full evaluation of host-guest interactions */
    double Predict()
    {
        size_t n_framework = ReplicaAtoms[0].size;

        size_t n_adsorbate = 0;
        for (size_t comp = 1; comp < ReplicaAtoms.size(); comp++)
            n_adsorbate += ReplicaAtoms[comp].size;

        size_t n_total = n_framework + n_adsorbate;

        static std::vector<double> xyz_total;
        static std::vector<double> xyz_framework;
        static std::vector<double> xyz_adsorbate;

        xyz_total.resize(3 * n_total);
        xyz_framework.resize(3 * n_framework);
        xyz_adsorbate.resize(3 * n_adsorbate);

        size_t counter = 0;
        for (size_t comp = 0; comp < ReplicaAtoms.size(); comp++)
        for (size_t i = 0; i < ReplicaAtoms[comp].size; i++)
        {
            xyz_total[3*counter + 0] = ReplicaAtoms[comp].pos[i].x;
            xyz_total[3*counter + 1] = ReplicaAtoms[comp].pos[i].y;
            xyz_total[3*counter + 2] = ReplicaAtoms[comp].pos[i].z;
            counter++;
        }

        for (size_t i = 0; i < n_framework; i++)
        {
            xyz_framework[3*i + 0] = ReplicaAtoms[0].pos[i].x;
            xyz_framework[3*i + 1] = ReplicaAtoms[0].pos[i].y;
            xyz_framework[3*i + 2] = ReplicaAtoms[0].pos[i].z;
        }

        counter = 0;
        for (size_t comp = 1; comp < ReplicaAtoms.size(); comp++)
        for (size_t i = 0; i < ReplicaAtoms[comp].size; i++)
        {
            xyz_adsorbate[3*counter + 0] = ReplicaAtoms[comp].pos[i].x;
            xyz_adsorbate[3*counter + 1] = ReplicaAtoms[comp].pos[i].y;
            xyz_adsorbate[3*counter + 2] = ReplicaAtoms[comp].pos[i].z;
            counter++;
        }

        double E_total_ev     = PredictFromSocket(xyz_total.data(), n_total);
        double E_framework_ev = PredictFromSocket(xyz_framework.data(), n_framework);
        double E_adsorbate_ev = PredictFromSocket(xyz_adsorbate.data(), n_adsorbate);

        return E_total_ev - E_framework_ev - E_adsorbate_ev;
    }

    /* Clean shutdown */
    void close_socket()
    {
        if (fd >= 0) {
            send_command("EXIT");
            close(fd);
            fd = -1;
        }
    }

    void Match_Element_PseudoAtom_with_model(PseudoAtomDefinitions& PseudoAtoms)
    {
        printf("------- MATCHING ELEMENTS WITH PSEUDO ATOM ELEMENT SYMBOLS -------\n");
        Match_Element_PseudoAtom_order.resize(PseudoAtoms.Symbol.size(), -1);
        for(size_t i = 0; i < PseudoAtoms.Symbol.size(); i++)
        {
        for(size_t j = 0; j < ElementSymbolUsed.size(); j++)
            if(PseudoAtoms.Symbol[i] == ElementSymbolUsed[j])
            {
            Match_Element_PseudoAtom_order[i] = j;
            printf("PseudoAtom Symbol[%zu]: %s, Socket Symbol[%zu]: %s, MATCHED\n", i, PseudoAtoms.Symbol[i].c_str(), j, ElementSymbolUsed[j].c_str());
            break;
            }
        }
    }

    double dot(double3 a, double3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    double matrix_determinant(double* x) //9*1 array
    {
        double m11 = x[0*3+0]; double m21 = x[1*3+0]; double m31 = x[2*3+0];
        double m12 = x[0*3+1]; double m22 = x[1*3+1]; double m32 = x[2*3+1];
        double m13 = x[0*3+2]; double m23 = x[1*3+2]; double m33 = x[2*3+2];
        double determinant = +m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
        return determinant;
    }

    void inverse_matrix(double* x, double **inverse_x)
    {
        double m11 = x[0*3+0]; double m21 = x[1*3+0]; double m31 = x[2*3+0];
        double m12 = x[0*3+1]; double m22 = x[1*3+1]; double m32 = x[2*3+1];
        double m13 = x[0*3+2]; double m23 = x[1*3+2]; double m33 = x[2*3+2];
        double determinant = +m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
        double* result = (double*) malloc(9 * sizeof(double));
        result[0] = +(m22 * m33 - m32 * m23) / determinant;
        result[3] = -(m21 * m33 - m31 * m23) / determinant;
        result[6] = +(m21 * m32 - m31 * m22) / determinant;
        result[1] = -(m12 * m33 - m32 * m13) / determinant;
        result[4] = +(m11 * m33 - m31 * m13) / determinant;
        result[7] = -(m11 * m32 - m31 * m12) / determinant;
        result[2] = +(m12 * m23 - m22 * m13) / determinant;
        result[5] = -(m11 * m23 - m21 * m13) / determinant;
        result[8] = +(m11 * m22 - m21 * m12) / determinant;
        *inverse_x = result;
    }

    __host__ double3 GetFractionalCoord(double* InverseCell, bool Cubic, double3 posvec)
    {
        double3 s = {0.0, 0.0, 0.0};
        s.x=InverseCell[0*3+0]*posvec.x + InverseCell[1*3+0]*posvec.y + InverseCell[2*3+0]*posvec.z;
        s.y=InverseCell[0*3+1]*posvec.x + InverseCell[1*3+1]*posvec.y + InverseCell[2*3+1]*posvec.z;
        s.z=InverseCell[0*3+2]*posvec.x + InverseCell[1*3+2]*posvec.y + InverseCell[2*3+2]*posvec.z;
        return s;
    }

    __host__ double3 GetRealCoordFromFractional(double* Cell, bool Cubic, double3 s)
    {
        double3 posvec = {0.0, 0.0, 0.0};
        posvec.x=Cell[0*3+0]*s.x+Cell[1*3+0]*s.y+Cell[2*3+0]*s.z;
        posvec.y=Cell[0*3+1]*s.x+Cell[1*3+1]*s.y+Cell[2*3+1]*s.z;
        posvec.z=Cell[0*3+2]*s.x+Cell[1*3+2]*s.y+Cell[2*3+2]*s.z;
        return posvec;
    }

    std::vector<std::string> split(const std::string txt, char ch)
    {
        size_t pos = txt.find(ch);
        size_t initialPos = 0;
        std::vector<std::string> strs{};

        // Decompose statement
        while (pos != std::string::npos) {

            std::string s = txt.substr(initialPos, pos - initialPos);
            if (!s.empty())
            {
                strs.push_back(s);
            }
            initialPos = pos + 1;

            pos = txt.find(ch, initialPos);
        }

        // Add the last one
        std::string s = txt.substr(initialPos, std::min(pos, txt.size()) - initialPos + 1);
        if (!s.empty())
        {
            strs.push_back(s);
        }

        return strs;
    }

    bool caseInSensStringCompare(const std::string& str1, const std::string& str2)
    {
        return str1.size() == str2.size() && std::equal(str1.begin(), str1.end(), str2.begin(), [](auto a, auto b) {return std::tolower(a) == std::tolower(b); });
    }

    void Split_Tab_Space(std::vector<std::string>& termsScannedLined, std::string& str)
    {
        if (str.find("\t", 0) != std::string::npos) //if the delimiter is tab
        {
        termsScannedLined = split(str, '\t');
        }
        else
        {
        termsScannedLined = split(str, ' ');
        }
    }

    void AllocateUCSpace(size_t comp)
    {
        UCAtoms[comp].pos   = (double3*) malloc(UCAtoms[comp].size * sizeof(double3));
        UCAtoms[comp].Type  = (size_t*)  malloc(UCAtoms[comp].size * sizeof(size_t));
    }

    void GenerateUCBox(double* SuperCell, int3 Ncell)
    {
        UCBox.Cell        = (double*) malloc(9 * sizeof(double));
        UCBox.InverseCell = (double*) malloc(9 * sizeof(double));
        for(size_t i = 0; i < 9; i++) UCBox.Cell[i] = SuperCell[i];
        UCBox.Cell[0] /= Ncell.x; UCBox.Cell[1]  = 0.0;     UCBox.Cell[2]  = 0.0;
        UCBox.Cell[3] /= Ncell.y; UCBox.Cell[4] /= Ncell.y; UCBox.Cell[5]  = 0.0;
        UCBox.Cell[6] /= Ncell.z; UCBox.Cell[7] /= Ncell.z; UCBox.Cell[8] /= Ncell.z;
        inverse_matrix(UCBox.Cell, &UCBox.InverseCell);
    }

    //Assuming the framework atoms are reproduced, and the order of atoms in a unit cell matches the order in the cif//
    //This assumes rigid framework//
    //Since it is framework (assuming all the atoms are DNN-used, consider them all)
    //This will gaurentee that UCAtoms.size = DNN_consider_size, not NAtoms or HostAtoms.Molsize;
    //Consider a TIP4P molecule, there is a fictional charge site that needs to be excluded//
    //So the DNN will consider 2*hydrogen + 1*oxygen, but not the fictional charge site//
    //TIP4P HostAtoms.Molsize = 4, here we want UCAtoms.size = 3//
    void CopyAtomsFromFirstUnitcell(Atoms& HostAtoms, size_t comp, int3 NSupercell, PseudoAtomDefinitions& PseudoAtoms, bool* ConsiderThisAdsorbateAtom)
    {
        size_t NAtoms = HostAtoms.Molsize / (NSupercell.x * NSupercell.y * NSupercell.z);
        if(HostAtoms.size % NAtoms != 0) throw std::runtime_error("SuperCell size cannot be divided by number of supercell atoms!!!!");
        UCAtoms[comp].size = NAtoms;
        //During the initialization phase, for adsorbate atoms, exclude those that are NOT considered in DNN.
        //Still assuming one adsorbate species, do it only for adsorbate
        if(comp != 0)
        for(size_t i = 0; i < NAtoms; i++)
            if(ConsiderThisAdsorbateAtom[i])
            DNN_Molsize += 1;

        if(comp != 0) UCAtoms[comp].size = DNN_Molsize;
        AllocateUCSpace(comp);

        size_t update_i = 0;
        for(size_t i = 0; i < NAtoms; i++)
        {
        if(comp != 0)
            if(!ConsiderThisAdsorbateAtom[i]) continue;
        UCAtoms[comp].pos[update_i]  = HostAtoms.pos[i];
        size_t SymbolIdx = Match_Element_PseudoAtom_order[HostAtoms.Type[i]];
        UCAtoms[comp].Type[update_i] = SymbolIdx;
        if(i < 5 || i > (NAtoms - 5)) printf("Component %zu, Atom %zu, xyz %f %f %f, Type %zu, SymbolIndex %zu\n", comp, i, UCAtoms[comp].pos[i].x, UCAtoms[comp].pos[i].y, UCAtoms[comp].pos[i].z, HostAtoms.Type[i], UCAtoms[comp].Type[i]);
        update_i ++;
        }
    }
    
    void ReplicateAtomsPerComponent(size_t comp, bool Allocate)
    {
        //Get Fractional positions//
        std::vector<double3>fpos;
        for(size_t i = 0; i < UCAtoms[comp].size; i++)
        {
        fpos.push_back(GetFractionalCoord(UCBox.InverseCell, UCBox.Cubic, UCAtoms[comp].pos[i]));
        }

        size_t NTotalCell = static_cast<size_t>(NReplicacell.x * NReplicacell.y * NReplicacell.z);
        double3 Shift = {(double)1/NReplicacell.x, (double)1/NReplicacell.y, (double)1/NReplicacell.z};
        if(NReplicacell.x % 2 == 0) throw std::runtime_error("Ncell in x needs to be an odd number (so that original unit cell sits in the center\n");
        if(NReplicacell.y % 2 == 0) throw std::runtime_error("Ncell in y needs to be an odd number (so that original unit cell sits in the center\n");
        if(NReplicacell.z % 2 == 0) throw std::runtime_error("Ncell in z needs to be an odd number (so that original unit cell sits in the center\n");
        int Minx = (NReplicacell.x - 1)/2 * -1; int Maxx = (NReplicacell.x - 1)/2;
        int Miny = (NReplicacell.y - 1)/2 * -1; int Maxy = (NReplicacell.y - 1)/2;
        int Minz = (NReplicacell.z - 1)/2 * -1; int Maxz = (NReplicacell.z - 1)/2;
        std::vector<int>xs; std::vector<int>ys; std::vector<int>zs;
        xs.push_back(0);
        ys.push_back(0);
        zs.push_back(0);
        for(int i = Minx; i <= Maxx; i++)
        if(i != 0)
            xs.push_back(i);
        for(int i = Miny; i <= Maxy; i++)
        if(i != 0)
            ys.push_back(i);
        for(int i = Minz; i <= Maxz; i++)
        if(i != 0)
            zs.push_back(i);

        if(Allocate)
        {
        ReplicaAtoms[comp].pos   = (double3*) malloc(NTotalCell * UCAtoms[comp].size * sizeof(double3));
        ReplicaAtoms[comp].Type  = (size_t*)  malloc(NTotalCell * UCAtoms[comp].size * sizeof(size_t));
        }
        size_t counter = 0;
        for(size_t a = 0; a < static_cast<size_t>(NReplicacell.x); a++)
        for(size_t b = 0; b < static_cast<size_t>(NReplicacell.y); b++)
            for(size_t c = 0; c < static_cast<size_t>(NReplicacell.z); c++)
            {
            int ix = xs[a];
            int jy = ys[b];
            int kz = zs[c];
            //printf("a: %zu, ix: %d, b: %zu, jy: %d, c: %zu, kz: %d\n", a, ix, b, jy, c, kz);
            double3 NCellID = {(double) ix, (double) jy, (double) kz};
            for(size_t i = 0; i < UCAtoms[comp].size; i++)
            {
                double3 temp = {fpos[i].x + NCellID.x, 
                                fpos[i].y + NCellID.y,
                                fpos[i].z + NCellID.z};
                double3 super_fpos = {temp.x * Shift.x, 
                                    temp.y * Shift.y, 
                                    temp.z * Shift.z};
                // Get real xyz from fractional xyz //
                double3 Replica_pos;
                Replica_pos.x = super_fpos.x*ReplicaBox.Cell[0]+super_fpos.y*ReplicaBox.Cell[3]+super_fpos.z*ReplicaBox.Cell[6];
                Replica_pos.y = super_fpos.x*ReplicaBox.Cell[1]+super_fpos.y*ReplicaBox.Cell[4]+super_fpos.z*ReplicaBox.Cell[7];
                Replica_pos.z = super_fpos.x*ReplicaBox.Cell[2]+super_fpos.y*ReplicaBox.Cell[5]+super_fpos.z*ReplicaBox.Cell[8];
                ReplicaAtoms[comp].pos[counter]   = Replica_pos;
                ReplicaAtoms[comp].Type[counter]  = UCAtoms[comp].Type[i];
                counter ++;
            }
            }
        ReplicaAtoms[comp].size = NTotalCell * UCAtoms[comp].size;
    }
    
    //For Single Unit cell atoms, we separate them into different components//
    //For replica, we also do that//
    void GenerateReplicaCells(bool Allocate)
    {
        size_t NComp = UCAtoms.size();
        size_t N_UCAtom = 0; for(size_t comp = 0; comp < NComp; comp++) N_UCAtom += UCAtoms[comp].size;
        if(Allocate)
        {
        ReplicaBox.Cell = (double*) malloc(9 * sizeof(double));
        ReplicaBox.InverseCell = (double*) malloc(9 * sizeof(double));
        for(size_t i = 0; i < 9; i++) ReplicaBox.Cell[i] = UCBox.Cell[i];
    
        ReplicaBox.Cell[0] *= NReplicacell.x; ReplicaBox.Cell[1] *= 0.0;            ReplicaBox.Cell[2] *= 0.0;
        ReplicaBox.Cell[3] *= NReplicacell.y; ReplicaBox.Cell[4] *= NReplicacell.y; ReplicaBox.Cell[5] *= 0.0;
        ReplicaBox.Cell[6] *= NReplicacell.z; ReplicaBox.Cell[7] *= NReplicacell.z; ReplicaBox.Cell[8] *= NReplicacell.z;

        printf("a: %f, b: %f, c: %f\n", ReplicaBox.Cell[0], ReplicaBox.Cell[4], ReplicaBox.Cell[8]);
        inverse_matrix(ReplicaBox.Cell, &ReplicaBox.InverseCell);
        }
        //Assuming Framework fixed//
        for(size_t comp = 0; comp < UCAtoms.size(); comp++)
        if(Allocate || comp != 0)
            ReplicateAtomsPerComponent(comp, Allocate);
    }
    
    void WrapSuperCellAtomIntoUCBox(size_t comp)
    {
        std::vector<double>Bonds; //For checking bond distances if the molecule is wrapped onto different sides of the box//
        std::vector<double3>newpos;
        for(size_t i = 0; i < UCAtoms[comp].size; i++)
        {
        double3 FPOS = GetFractionalCoord(UCBox.InverseCell, UCBox.Cubic, UCAtoms[comp].pos[i]);
        //printf("New Atom %zu, fxyz: %f %f %f\n", i, FPOS.x, FPOS.y, FPOS.z);
        double3 FLOOR = {(double)floor(FPOS.x), (double)floor(FPOS.y), (double)floor(FPOS.z)};
        double3 New_fpos = {FPOS.x - FLOOR.x,
                            FPOS.y - FLOOR.y,
                            FPOS.z - FLOOR.z};

        newpos.push_back(New_fpos);
        
        if(i == 0) continue;
        double3 dist = {UCAtoms[comp].pos[i].x - UCAtoms[comp].pos[i - 1].x, 
                        UCAtoms[comp].pos[i].y - UCAtoms[comp].pos[i - 1].y,
                        UCAtoms[comp].pos[i].z - UCAtoms[comp].pos[i - 1].z};
        double dsq = dot(dist, dist);
        Bonds.push_back(dsq);     
        }
        for(size_t i = 0; i < UCAtoms[comp].size; i++)
        {
        double3 Real_Pos = GetRealCoordFromFractional(UCBox.Cell, UCBox.Cubic, newpos[i]);
        UCAtoms[comp].pos[i] = Real_Pos;
        }
    }

    //This function is called after the position of the trial adsorbate molecule is prepared in UCAtoms//
    double MCEnergyWrapper(size_t comp, bool Initialize, double DNNEnergyConversion)
    {
        WrapSuperCellAtomIntoUCBox(comp);
        GenerateReplicaCells(Initialize);

        double DNN_E = Predict();
        //This generates the unit of eV, convert to 10J/mol.
        //https://www.weizmann.ac.il/oc/martin/tools/hartree.html
        return DNN_E * DNNEnergyConversion;
    }

    };
