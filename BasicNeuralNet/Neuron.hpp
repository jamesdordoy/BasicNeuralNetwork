//
//  Neuron.hpp
//  BasicNeuralNet
//
//  Created by James Dordoy on 14/10/2019.
//  Copyright © 2019 James Dordoy. All rights reserved.
//

#ifndef Neuron_hpp
#define Neuron_hpp

#include <stdio.h>
#include <vector>

#endif /* Neuron_hpp */

class Neuron
{
public:
    Neuron(); //unsigned numberOfOutputs
private:
    double outputValue;
    vector<Connection> outputWeights;
};
