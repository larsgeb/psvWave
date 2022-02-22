Typical FWI workflow
====================


To perform full-waveform inversion, the following actions must be performed:

1. Set some observed data (we did this in the previous cell)
2. Forward simulate all sources
3. Calculate the misfit
4. Calculate the adjoint sources
5. Empty the kernels (from any previous calculation)
6. Adjoint simulate all sources
7. Transform the sensitivity kernels (in Lamé's paramters) to velocity

In Notebook 2, these actions are described. If you simply want a class that combines
everything, use the following code, given you have a fdModel instance stored in 
`solver`::

    class _fwi:

        last_model = numpy.empty_like(solver.get_model_vector())
        misfit = None
        g = None
        smoothing = None

        def __init__(self):
            pass

        def misfit(self, m):

            if numpy.allclose(m, self.last_model):
                return self.misfit
            else:
                _ = self.grad(m)
                return self.misfit

        def grad(self, m):

            if numpy.allclose(m, self.last_model):
                return self.g
            else:
                solver.set_model_vector(m)

                # Simulate forward
                for i_shot in range(solver.n_shots):
                    solver.forward_simulate(i_shot, omp_threads_override=6)

                # Calculate misfit and adjoint sources
                solver.calculate_l2_misfit()
                solver.calculate_l2_adjoint_sources()
                self.misfit = solver.misfit

                # Simulate adjoint
                solver.reset_kernels()
                for i_shot in range(solver.n_shots):
                    solver.adjoint_simulate(i_shot, omp_threads_override=6)
                solver.map_kernels_to_velocity()

                g = solver.get_gradient_vector()

                if self.smoothing is not None:
                    g = numpy.hstack(
                        [
                            gaussian_filter(
                                igs.reshape((60, 180)), sigma=self.smoothing
                            ).flatten()
                            for igs in numpy.split(g, 3)
                        ]
                    )

                self.g = g

                self.last_model = m

                return self.g
