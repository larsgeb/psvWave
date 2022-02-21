import psvWave
import numpy


def test_copy():
    model = psvWave.fdModel(
        "tests/test_configurations/default_testing_configuration.ini"
    )

    model2: psvWave.fdModel = model.copy()


def test_copy_modify():
    model1 = psvWave.fdModel(
        "tests/test_configurations/default_testing_configuration.ini"
    )

    model2: psvWave.fdModel = model1.copy()

    model1.set_model_vector(model1.get_model_vector() + 1)

    model2.set_model_vector(model2.get_model_vector() - 1)

    assert numpy.any(model2.get_model_vector() != model1.get_model_vector())
