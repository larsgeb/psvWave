import psvWave
import numpy


def test_descent():
    model = psvWave.fdModel(
        "../../tests/test_configurations/default_testing_configuration.ini"
    )

    # Create target model ---------------------------------------------------------

    # Get the coordinates of every grid point
    IX, IZ = model.get_coordinates(True)
    extent = model.get_extent(True)
    # Get the associated parameter fields
    vp, vs, rho = model.get_parameter_fields()

    vp_starting = vp
    vs_starting = vs
    rho_starting = rho

    x_middle = (IX.max() + IX.min()) / 2
    z_middle = (IZ.max() + IZ.min()) / 2

    circle = ((IX - x_middle) ** 2 + (IZ - z_middle) ** 2) ** 0.5 < 15
    vs = vs * (1 - 0.1 * circle)
    vp = vp * (1 - 0.1 * circle)

    vp_target = vp
    vs_target = vs
    rho_target = rho

    model.set_parameter_fields(vp_target, vs_target, rho_target)

    # Create true data ------------------------------------------------------------

    for i_shot in range(model.n_shots):
        model.forward_simulate(i_shot, omp_threads_override=6)

    # Cheating of course, as this is synthetically generated data.
    ux_obs, uz_obs = model.get_synthetic_data()

    # numpy.random.seed(0)
    # std = 10.0
    # ux_obs += std * numpy.random.randn(*ux_obs.shape)
    # uz_obs += std * numpy.random.randn(*uz_obs.shape)

    model.set_observed_data(ux_obs, uz_obs)

    # Reverting the model to the starting model -----------------------------------

    vp = vp_starting
    vs = vs_starting
    rho = rho_starting

    model.set_parameter_fields(vp_starting, vs_starting, rho_starting)

    for i_shot in range(model.n_shots):
        model.forward_simulate(i_shot, omp_threads_override=6)

    ux, uz = model.get_synthetic_data()
    ux_obs, uz_obs = model.get_observed_data()

    max_waveform = max(ux.max(), uz.max(), ux_obs.max(), uz.max()) / 2

    m_ux_obs = ux_obs.copy()
    m_ux = ux.copy()

    for i in range(ux_obs.shape[1]):
        m_ux_obs[0, i:, :] += max_waveform
        m_ux[0, i:, :] += max_waveform

    # Perform adjoint simulation --------------------------------------------------

    model.calculate_l2_misfit()

    print(f"Data misfit: {model.misfit:.2f}")

    model.calculate_l2_adjoint_sources()
    model.reset_kernels()
    for i_shot in range(model.n_shots):
        model.adjoint_simulate(i_shot, omp_threads_override=6)
    model.map_kernels_to_velocity()

    g_vp, g_vs, g_rho = model.get_kernels()

    extrema = numpy.abs(g_vp).max(), numpy.abs(g_vs).max(), numpy.abs(g_rho).max()

    extent = (extent[0], extent[1], extent[3], extent[2])

    gradients = [g_vp, g_vs, g_rho]

    # Start iterating -------------------------------------------------------------

    m = model.get_model_vector()

    print("Starting gradient descent")

    fields_during_iteration = []

    iterations = 3

    misfits = []

    try:
        for i in range(iterations):

            g = model.get_gradient_vector()

            m -= 0.25 * g
            model.set_model_vector(m)

            fields_during_iteration.append(list(model.get_parameter_fields()))

            # Simulate forward
            for i_shot in range(model.n_shots):
                model.forward_simulate(i_shot, omp_threads_override=6)

            # Calculate misfit and adjoint sources
            model.calculate_l2_misfit()
            model.calculate_l2_adjoint_sources()
            print(f"Data misfit: {model.misfit:.2f}")
            misfits.append(model.misfit)

            # Simulate adjoint
            model.reset_kernels()
            for i_shot in range(model.n_shots):
                model.adjoint_simulate(i_shot, omp_threads_override=6)
            model.map_kernels_to_velocity()
    except KeyboardInterrupt:
        m = model.get_model_vector()
        iterations = i

    print(f"Misfit at start {misfits[0]}")
    print(f"Misfit at end   {misfits[-1]}")

    assert misfits[0] > misfits[-1]

