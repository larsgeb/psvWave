//
// Created by Lars Gebraad on 28/12/18.
//

// Includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include "fdWaveModel.h"

int main() {

    auto *model = new fdWaveModel();

    int source = 0;

    (*model).forwardSimulate(true, true, source);

    std::ofstream rec;
    rec.open("rec.txt");

    for (int iRec = 0; iRec < (*model).nr; ++iRec) {
        rec << std::endl;
        for (const auto &rtf_it : (*model).rtf_ux[source][iRec]) {
            rec << rtf_it << " ";
        }
    }
    rec.close();



}

