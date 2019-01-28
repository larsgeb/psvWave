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

    (*model).forwardSimulate(true, true, 1);
    (*model).forwardSimulate(true, true, 0);


    (*model).write_receivers();


//    std::ofstream rec;
//    rec.open("rec.txt");
//    for (int iRec = 0; iRec < (*model).nr; ++iRec) {
//        rec << std::endl;
//        for (const auto &rtf_it : (*model).rtf_ux[source][iRec]) {
//            rec << rtf_it << " ";
//        }
//    }
//    rec.close();



}

