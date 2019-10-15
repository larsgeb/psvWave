#include <fstream>
#include <iostream>
#include "ext/forward-virieux/src/fdWaveModel.h"
using namespace std;

int main(int argc, char *argv[]) {
    fdWaveModel *model = new fdWaveModel("loop_config_fd.ini");

    model->load_model("uloop/de_target.txt", "uloop/vp_target.txt", "uloop/vs_target.txt", false);

    model->run_model(false, false);
    model->write_receivers("");
    model->load_receivers(false);

    // model->set_model_vector(model->load_vector("model.txtlast", true));
    model->set_model_vector(model->get_model_vector() * 1.01);

    model->run_model(false, true);

    auto misfit = model->misfit;

    std::cout << "Original misfit " << misfit << std::endl;

    auto model_vector = model->get_model_vector();
    auto gradient_vector = model->get_gradient_vector();

    auto dotproduct = model_vector.dot(gradient_vector);

    // Gradient descent
    for (int exponent = -5; exponent < 0; exponent++) {
        double epsilon = pow(10.0, double(exponent));
        std::cout << std::endl << "epsilon " << epsilon << " exp " << exponent << std::endl;
        model->set_model_vector(model_vector * (1.0 + epsilon));
        model->run_model(false, false);
        std::cout << "New misfit      " << model->misfit << std::endl;
        std::cout << "Delta           " << (model->misfit - misfit) << std::endl;
        std::cout << "Predicted       " << epsilon * (dotproduct) << std::endl;
        std::cout << "Relative error  "
                  << ((model->misfit - misfit) - epsilon * (dotproduct)) / (model->misfit - misfit) << std::endl;
    }

    // Try a gradient descent.
    model->load_model("uloop/de_starting.txt", "uloop/vp_starting.txt", "uloop/vs_starting.txt", false);

    model->run_model(false, true);
    misfit = model->misfit;
    std::cout << std::endl << "gradient descent" << std::endl;
    std::cout << "misfit " << misfit << std::endl;

    for (int i = 0; i < 1000; i++) {
        gradient_vector = model->get_gradient_vector();
        model->set_model_vector(model->get_model_vector() - 1e-10 * gradient_vector);
        model->run_model(false, true);
        if (model->misfit <= misfit) {
            misfit = model->misfit;
            std::cout << "new misfit " << misfit << std::endl;

            std::ofstream file("model" + zero_pad_number(i, 5) + ".txt");
            if (file.is_open()) {
                file << model->get_model_vector() << '\n';
            }
            file.close();

        } else {
            std::cout << "Terminating. last misfit " << model->misfit << std::endl;
            exit(0);
        }
    }
}