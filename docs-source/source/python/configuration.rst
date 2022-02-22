Configuration files
===================

The configuration files are what creates a wave eqaution solver in psvWave. In contains
all information on sources, geometry, etc. In Python, it is easiest to simply call the
dictionary and edit values in there.::
    
    
    import psvWave
    settings = psvWave.get_dictionary()

    settings ["domain"]["nt"] = 12000
    settings["sources"]["which_source_to_fire_in_which_shot"] = [[0]]

    psvWave.write_dictionary("solver_1.ini", settings)

    solver = psvWave.fdModel("solver_1.ini")
    
Note that the actual extension is not important, so go crazy.

One can also write the conf file outside of python, and load it later. A conf.ini file
structure is given below. All parameters should be self-explanatory with the help of the
API reference. If not, please raise an 
`issue on GitHub <https://github.com/larsgeb/psvWave/issues>`_.::

    [domain]
    nt = 8000;                              int
    nx_inner = 200;                         int
    nz_inner = 100;                         int
    nx_inner_boundary = 10;                 int, defines inner limits in which to compute kernels. Limits wavefield storage and computation burden.
    nz_inner_boundary = 20;                 int, defines inner limits in which to compute kernels. Limits wavefield storage and computation burden.
    dx = 1.249;                             float
    dz = 1.249;                             float
    dt = 0.00025;                           float

    [boundary]
    np_boundary = 25;      int
    np_factor = 0.015;      float

    [medium]; Default values for the simulated models if none are loaded
    scalar_rho = 1500.0;    float
    scalar_vp = 2000.0;     float
    scalar_vs = 800.0;      float

    [sources]
    peak_frequency = 50.0;                  float
    n_sources = 4;                          int
    n_shots = 1;                            int
    source_timeshift = 0.005;
    delay_cycles_per_shot = 24; // on time axis: 24/f
    moment_angles = {90, 180, 90, 180} ;
    ix_sources = {25, 75, 125, 175};
    iz_sources = {10, 10, 10, 10};
    which_source_to_fire_in_which_shot = {{0, 1, 2, 3}};

    [receivers]
    nr = 19; 
    ix_receivers = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190}; !!
    iz_receivers = {90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90}; !!

    [inversion]
    snapshot_interval = 10; int, snapshots of forward wavefield to store.

    [basis]
    npx = 1
    npz = 1

    [output]
    observed_data_folder = .
    stf_folder = .