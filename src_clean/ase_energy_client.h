#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <errno.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Config-type constants for socket routing.
// These are sent as a 4-byte int32 BEFORE the cell matrix in every POSDATA
// exchange so the Python server can unambiguously identify which symbol list
// to use, regardless of atom count.
//
//   CONFIG_FRAMEWORK (0)   : framework-only configuration
//   N   (1, 2, ...)        : adsorbate-only for component N
//   100+N (101, 102, ...)  : total (framework + adsorbate component N)
enum ConfigType : int32_t {
    CONFIG_FRAMEWORK = 0,
    // adsorbate-only:  use ads_comp directly (1, 2, ...)
    // total:           use 100 + ads_comp (101, 102, ...)
};

struct Socket
{
    Boxsize UCBox;
    std::vector<Atoms> UCAtoms;
    std::vector<std::string> ElementSymbolUsed;
    std::vector<int>Match_Element_PseudoAtom_order; //length = # of PseudoAtoms, value stored = order in the Socket//

    char socket_path[108] = "/tmp/ase_ipi_socket";  // UNIX path limit
    int    fd     = -1;   // UNIX socket file descriptor
    size_t natoms = 0;    // number of atoms sent in last call

    size_t DNN_Molsize = 0; //Atom size to be considered for DNN, since there might be fictional atom sites for a classical sim molecule

    size_t nstep = 0;

    bool handshake_done = false;

    /* Constructor-style init */
    void init(const char *path, int n_atoms)
    {
        fd = -1;
        natoms = n_atoms;
        strncpy(socket_path, path, sizeof(socket_path) - 1);
        socket_path[sizeof(socket_path) - 1] = '\0';
    }

    void send_header(const char *msg)
    {
        char header[12];
        memset(header, ' ', 12);
        strncpy(header, msg, std::min((size_t)12, strlen(msg)));
        write_all(header, 12);
    }
    
    // ADD THIS: Send int32 in network byte order  
    void send_int32(int32_t value)
    {
        int32_t net_val = htonl(value);
        write_all(&net_val, sizeof(int32_t));
    }
    
    // ADD THIS: Initial handshake with ASE
    int do_ipi_handshake()
    {
        // 1. Receive STATUS
        char header[12];
        if (read_all(header, 12) != 12) {
            fprintf(stderr, "Failed to read STATUS header\n");
            return -1;
        }
        
        if (strncmp(header, "STATUS", 6) != 0) {
            fprintf(stderr, "Expected STATUS, got: %.12s\n", header);
            return -1;
        }
        
        // 2. Send NEEDINIT
        send_header("NEEDINIT");
        
        // 3. Receive INIT
        if (read_all(header, 12) != 12) {
            fprintf(stderr, "Failed to read INIT header\n");
            return -1;
        }
        
        if (strncmp(header, "INIT", 4) != 0) {
            fprintf(stderr, "Expected INIT, got: %.12s\n", header);
            return -1;
        }
        
        // 4. Receive cell matrices (discard for now)
        double cell[9], inv_cell[9];
        read_all(cell, sizeof(double) * 9);
        read_all(inv_cell, sizeof(double) * 9);
        
        printf("i-PI handshake complete\n");
        return 0;
    }   

    // Send species map immediately after handshake.
    // Binary format (all int32 in network byte order):
    //   "SPECIESMAP" header (12-byte padded ASCII)
    //   int32  n_species
    //   int32  n_fw       (atom count in UCAtoms[0])
    //   for each species: int32 index, int32 len, len bytes symbol
    void send_species_map()
    {
        send_header("SPECIESMAP");
        send_int32((int32_t)ElementSymbolUsed.size());
        send_int32((int32_t)UCAtoms[0].size);   // n_fw atoms
        for (size_t i = 0; i < ElementSymbolUsed.size(); i++) {
            send_int32((int32_t)i);
            int32_t len = (int32_t)ElementSymbolUsed[i].size();
            send_int32(len);
            write_all(ElementSymbolUsed[i].c_str(), len);
        }
        printf("Sent species map: %zu species, n_fw=%zu\n",
               ElementSymbolUsed.size(), UCAtoms[0].size);
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

        printf("Connecting to CHGNet server at: %s\n", socket_path);

        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            perror("connect");
            close(fd);
            fd = -1;
            return -1;
        }

        printf("Connected successfully\n");
    
        if (do_ipi_handshake() < 0) {
            close(fd);
            fd = -1;
            return -1;
        }
        
        handshake_done = true;
        send_species_map();
        PrimeFrameworkCache();    // prime E_fw cache in Python server

        return 0;
    }

    /* Send framework atoms to Python so it can compute and cache E_fw.
       Called at end of connect_socket() after send_species_map(). */
    void PrimeFrameworkCache()
    {
        size_t n_fw = UCAtoms[0].size;
        std::vector<double>  xyz_fw(3 * n_fw);
        std::vector<int32_t> types_fw(n_fw);
        for (size_t i = 0; i < n_fw; i++) {
            xyz_fw[3*i+0] = UCAtoms[0].pos[i].x;
            xyz_fw[3*i+1] = UCAtoms[0].pos[i].y;
            xyz_fw[3*i+2] = UCAtoms[0].pos[i].z;
            types_fw[i]   = (int32_t)UCAtoms[0].Type[i];
        }
        double E_fw_ev = PredictFromSocket(
            xyz_fw.data(), types_fw.data(), n_fw,
            (int32_t)CONFIG_FRAMEWORK, 0);
        printf("Python E_fw cache primed: %.6f eV\n", E_fw_ev);
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

    /* Send positions + types.
       Wire format after STATUS/READY/POSDATA handshake:
         int32   config_type    routing tag
         9×f64   cell
         9×f64   inv_cell       (zeroed — server uses PBC via cell)
         int32   n_mol          adsorbate molecule count (0 for fw/ads-only calls)
         int32   n_atoms
         N×int32 types          species index per atom
         3N×f64  xyz            Cartesian coordinates (Å)
    */
    int send_positions(const double *xyz, const int32_t *types,
                       size_t n_atoms, int32_t config_type, int32_t n_mol)
    {
        char header[12];

        // 1. Wait for STATUS
        read_all(header, 12);
        if (strncmp(header, "STATUS", 6) != 0) {
            fprintf(stderr, "Expected STATUS, got: %.12s\n", header);
            return -1;
        }

        // 2. Send READY
        send_header("READY");

        // 3. Wait for POSDATA request
        read_all(header, 12);
        if (strncmp(header, "POSDATA", 7) != 0) {
            fprintf(stderr, "Expected POSDATA, got: %.12s\n", header);
            return -1;
        }

        // 4. config_type routing tag
        send_int32(config_type);

        // 5. Cell matrix (9 doubles)
        write_all(UCBox.Cell, sizeof(double) * 9);

        // 6. Inverse cell (9 doubles, zeroed)
        double inv_cell[9] = {0};
        write_all(inv_cell, sizeof(double) * 9);

        // 7. n_mol
        send_int32(n_mol);

        // 8. n_atoms
        send_int32((int32_t)n_atoms);

        // 9. Types array (n_atoms × int32, network byte order)
        std::vector<int32_t> types_net(n_atoms);
        for (size_t i = 0; i < n_atoms; i++)
            types_net[i] = htonl((uint32_t)types[i]);
        write_all(types_net.data(), sizeof(int32_t) * n_atoms);

        // 10. Positions (3*n_atoms doubles)
        write_all(xyz, sizeof(double) * 3 * n_atoms);

        // 11. Wait for STATUS
        read_all(header, 12);
        if (strncmp(header, "STATUS", 6) != 0) {
            fprintf(stderr, "Expected STATUS, got: %.12s\n", header);
            return -1;
        }

        // 12. Send HAVEDATA
        send_header("HAVEDATA");

        // 13. Wait for GETFORCE
        read_all(header, 12);
        if (strncmp(header, "GETFORCE", 8) != 0) {
            fprintf(stderr, "Expected GETFORCE, got: %.12s\n", header);
            return -1;
        }

        return 0;
    }

    int receive_energy(double *energy_ev)
    {
        char header[12];
        
        // 1. Read FORCEREADY header (12 bytes!)
        if (read_all(header, 12) != 12) {
            perror("read FORCEREADY");
            return -1;
        }
        
        if (strncmp(header, "FORCEREADY", 10) != 0) {
            fprintf(stderr, "Expected FORCEREADY, got: %.12s\n", header);
            return -1;
        }
        
        // 2. Read energy (1 double)
        if (read_all(energy_ev, sizeof(double)) != sizeof(double)) {
            perror("read energy");
            return -1;
        }
        
        // 3. Read natoms (int32)
        int32_t recv_natoms;
        if (read_all(&recv_natoms, sizeof(int32_t)) != sizeof(int32_t)) {
            perror("read natoms");
            return -1;
        }
        recv_natoms = ntohl(recv_natoms);  // Convert from network byte order
        
        // 4. Read forces (3*natoms doubles)
        std::vector<double> forces(3 * recv_natoms);
        if (read_all(forces.data(), sizeof(double) * 3 * recv_natoms) <= 0) {
            perror("read forces");
            return -1;
        }
        
        // 5. Read virial (9 doubles)
        double virial[9];
        if (read_all(virial, sizeof(double) * 9) <= 0) {
            perror("read virial");
            return -1;
        }
        
        // 6. Read extras length
        int32_t extras_len;
        if (read_all(&extras_len, sizeof(int32_t)) != sizeof(int32_t)) {
            perror("read extras_len");
            return -1;
        }
        extras_len = ntohl(extras_len);
        
        // 7. Read extras if present
        if (extras_len > 0) {
            std::vector<char> extras(extras_len);
            read_all(extras.data(), extras_len);
        }
        
        return 0;
    }

    /* Send coords + types and receive single energy scalar. */
    double PredictFromSocket(const double* xyz, const int32_t* types,
                             size_t n_atoms, int32_t config_type, int32_t n_mol)
    {
        if (fd < 0) {
            fprintf(stderr, "Socket not connected — attempting auto-connect\n");
            if (connect_socket() < 0) {
                fprintf(stderr, "FATAL: Cannot connect to ML server\n");
                exit(EXIT_FAILURE);
            }
        }

        natoms = n_atoms;

        printf("[CLIENT] POSDATA natoms=%zu config_type=%d n_mol=%d\n",
               natoms, (int)config_type, (int)n_mol);
        for (size_t i = 0; i < std::min(natoms, (size_t)5); i++)
            printf("  atom %zu: %.5f %.5f %.5f type=%d\n",
                   i, xyz[3*i], xyz[3*i+1], xyz[3*i+2], (int)types[i]);
        fflush(stdout);

        send_positions(xyz, types, n_atoms, config_type, n_mol);

        double energy_ev;
        receive_energy(&energy_ev);
        return energy_ev;
    }

    /* Full evaluation of host-guest interaction for one trial adsorbate molecule.
       Python returns HG = E(fw+ads) - E_fw_cached - E(ads) directly. */
    double Predict(size_t ads_comp = 1)
    {
        size_t n_framework = UCAtoms[0].size;
        size_t n_adsorbate = UCAtoms[ads_comp].size;
        size_t n_total     = n_framework + n_adsorbate;

        std::vector<double>  xyz_total(3 * n_total);
        std::vector<int32_t> types_total(n_total);
        size_t counter = 0;

        for (size_t i = 0; i < n_framework; i++)
        {
            xyz_total[3*counter + 0] = UCAtoms[0].pos[i].x;
            xyz_total[3*counter + 1] = UCAtoms[0].pos[i].y;
            xyz_total[3*counter + 2] = UCAtoms[0].pos[i].z;
            types_total[counter]     = (int32_t)UCAtoms[0].Type[i];
            counter++;
        }
        for (size_t i = 0; i < n_adsorbate; i++)
        {
            xyz_total[3*counter + 0] = UCAtoms[ads_comp].pos[i].x;
            xyz_total[3*counter + 1] = UCAtoms[ads_comp].pos[i].y;
            xyz_total[3*counter + 2] = UCAtoms[ads_comp].pos[i].z;
            types_total[counter]     = (int32_t)UCAtoms[ads_comp].Type[i];
            counter++;
        }

        // Python returns HG = E(fw+ads) - E_fw_cached - E(ads) directly
        double HG_ev = PredictFromSocket(xyz_total.data(), types_total.data(),
                                         n_total, (int32_t)(100 + ads_comp), 1);
        printf("ML HG_ev = %.6f eV\n", HG_ev);
        return HG_ev;
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
        printf("UCBox cell matrix (Angstrom):\n");
        printf("  [ %12.6f  %12.6f  %12.6f ]\n", UCBox.Cell[0], UCBox.Cell[1], UCBox.Cell[2]);
        printf("  [ %12.6f  %12.6f  %12.6f ]\n", UCBox.Cell[3], UCBox.Cell[4], UCBox.Cell[5]);
        printf("  [ %12.6f  %12.6f  %12.6f ]\n", UCBox.Cell[6], UCBox.Cell[7], UCBox.Cell[8]);
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
        {
            size_t dnn_size = 0;
            for(size_t i = 0; i < NAtoms; i++)
                if(ConsiderThisAdsorbateAtom[i])
                    dnn_size++;
            DNN_Molsize = dnn_size;          // store for any other readers, but do NOT accumulate across components
            UCAtoms[comp].size = dnn_size;   // use local count directly
        }
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

    // Wrap a single Cartesian position into the primary unit cell [0,1) fractional range.
    double3 WrapPositionIntoUCBox(double3 pos)
    {
        double3 fpos = GetFractionalCoord(UCBox.InverseCell, UCBox.Cubic, pos);
        double3 flr  = {floor(fpos.x), floor(fpos.y), floor(fpos.z)};
        double3 nfpos = {fpos.x - flr.x, fpos.y - flr.y, fpos.z - flr.z};
        return GetRealCoordFromFractional(UCBox.Cell, UCBox.Cubic, nfpos);
    }

    // Called from PATCH_SOCKET_FXNMAIN (DNN_Prediction_Total).
    // Sends fw + all N adsorbate molecules in a single socket call.
    // Python returns HG = E(fw+N_ads) - E_fw_cached - E(N_ads together).
    // C++ multiplies by DNNEnergyConversion.
    //
    // host_positions : HostSystem[ads_comp].pos  (full simulation box coords, host ptr)
    // full_molsize   : Moleculesize[ads_comp]     (including fictional charge sites)
    // consider_atom  : ConsiderThisAdsorbateAtom  (length = full_molsize)
    // Returns energy already converted to gRASPA internal units (10 J/mol).
    double PredictTotal(size_t ads_comp, size_t n_mol, size_t full_molsize,
                        double3* host_positions, bool* consider_atom,
                        double DNNEnergyConversion)
    {
        if (n_mol == 0) return 0.0;

        size_t n_fw        = UCAtoms[0].size;
        size_t mol_size    = DNN_Molsize;            // atoms per molecule after filtering
        size_t n_ads_total = n_mol * mol_size;
        size_t n_total     = n_fw + n_ads_total;

        std::vector<double>  xyz(3 * n_total);
        std::vector<int32_t> types(n_total);

        // Framework atoms (pre-computed, stable)
        size_t counter = 0;
        for (size_t i = 0; i < n_fw; i++) {
            xyz[3*counter+0] = UCAtoms[0].pos[i].x;
            xyz[3*counter+1] = UCAtoms[0].pos[i].y;
            xyz[3*counter+2] = UCAtoms[0].pos[i].z;
            types[counter]   = (int32_t)UCAtoms[0].Type[i];
            counter++;
        }

        // All adsorbate molecules — wrap each atom into UCBox
        for (size_t i = 0; i < n_mol; i++) {
            size_t type_i = 0;
            for (size_t j = 0; j < full_molsize; j++) {
                if (!consider_atom[j]) continue;
                size_t atom_idx  = i * full_molsize + j;
                double3 wrapped  = WrapPositionIntoUCBox(host_positions[atom_idx]);
                xyz[3*counter+0] = wrapped.x;
                xyz[3*counter+1] = wrapped.y;
                xyz[3*counter+2] = wrapped.z;
                types[counter]   = (int32_t)UCAtoms[ads_comp].Type[type_i];
                counter++;
                type_i++;
            }
        }

        int32_t config_type_val = (int32_t)(100 + ads_comp);
        // Python returns HG = E(fw+N_ads) - E_fw_cached - E(N_ads together)
        double HG_ev = PredictFromSocket(xyz.data(), types.data(), n_total,
                                         config_type_val, (int32_t)n_mol);

        printf("ML PredictTotal: n_mol=%zu  HG=%.6f eV\n", n_mol, HG_ev);

        return HG_ev * DNNEnergyConversion;
    }

    //This function is called after the position of the trial adsorbate molecule is prepared in UCAtoms//
    double MCEnergyWrapper(size_t comp, bool Initialize, double DNNEnergyConversion)
    {
        WrapSuperCellAtomIntoUCBox(comp);

        double DNN_E = Predict(comp);
        //This generates the unit of eV, convert to 10J/mol.
        //https://www.weizmann.ac.il/oc/martin/tools/hartree.html
        return DNN_E * DNNEnergyConversion;
    }

    };
