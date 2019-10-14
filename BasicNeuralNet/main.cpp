//
//  main.cpp
//  BasicNeuralNet
//
//  Created by James Dordoy on 14/10/2019.
//  Copyright © 2019 James Dordoy. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

//Generic Function
void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

//Crap code for reading file
class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}


//Connection / Synapx
struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned numberOfOutputs, unsigned index);
    void setOutputValue(double value) { outputValue = value; };
    double getOutputValue() const { return outputValue; };
    void feedForward(const Layer &previousLayer);
    
    void calculateOutputGradients(double targetValue);
    void calculateHiddenGradidents(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
private:
    static double tranferFunction(double x);
    static double tranferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double outputValue;
    vector<Connection> outputWeights;
    unsigned index;
    double gradient;
    
    static double eta;
    static double alpha;
};

double Neuron::eta = 0.15; //Overal net learning rate

double Neuron::alpha = 0.5; //Momentum, multiplyer of last deltaWeight

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // In the Neurons in the preceding layer
    
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        
        double oldDeltaWeight = neuron.outputWeights[index].deltaWeight;
        
        double newDeltaWeight =
            // Individual input, magnifed by the gradient and training rate
            eta
            * neuron.getOutputValue()
            * gradient
            // Also add momentum = a fraction of the previous delta weigth
            + alpha
            * oldDeltaWeight;
        
        neuron.outputWeights[index].deltaWeight = newDeltaWeight;
        neuron.outputWeights[index].weight += newDeltaWeight;
    }
}


double Neuron::sumDOW(const Layer &nextLayer) const
{

    double sum = 0.0;
    
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += outputWeights[n].weight * nextLayer[n].gradient;
    }
    
    return sum;
};

void Neuron::calculateHiddenGradidents(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    
    gradient = dow * Neuron::tranferFunctionDerivative(outputValue);
}

void Neuron::calculateOutputGradients(double targetValue)
{
    double delta = targetValue - outputValue;
    gradient = delta * Neuron::tranferFunctionDerivative(outputValue);
};

double Neuron::tranferFunction(double x)
{
    return tanh(x);
};

double Neuron::tranferFunctionDerivative(double x)
{
    return 1.0 - x * x;
};

void Neuron::feedForward(const Layer &previousLayer)
{
    double sum = 0.0;
    
    // Sum previous layer neurons
    for (unsigned n = 0; n < previousLayer.size(); ++n) {
        sum += previousLayer[n].getOutputValue() *
            previousLayer[n].outputWeights[index].weight;
    }
    
    outputValue = Neuron::tranferFunction(sum);
};

Neuron::Neuron(unsigned numberOfOutputs, unsigned index)
{
    for (unsigned c = 0; c < numberOfOutputs; c++) {
        outputWeights.push_back(Connection());
        outputWeights.back().weight = randomWeight();
    }
    
    this->index = index;
};

class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backPropergate(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return recentAverageError; }
    
private:
    vector<Layer> layers; //layers_[layerNum][neuronNum]
    double error;
    double recentAverageError;
    double recentAverageSmoothingFactor;
};

//Prototype method
Net::Net(const vector<unsigned> &topology)
{
    unsigned long numberOfLayers = topology.size();
    
    for (unsigned layerNum = 0; layerNum < numberOfLayers; ++layerNum) {
        
        layers.push_back(Layer());
        unsigned numberOfOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        
        //We have added a new layer now we need to fill it with neurons
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            //Get last element
            layers.back().push_back(Neuron(numberOfOutputs, neuronNum));
        }
        
        layers.back().back().setOutputValue(1.0);
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == layers[0].size() - 1);
    
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        layers[0][i].setOutputValue(inputVals[i]);
    }
    
    for (unsigned layerNumber = 1; layerNumber < layers.size(); ++layerNumber) {
        
        Layer &previousLayer = layers[layerNumber - 1];
        for (unsigned n = 0; n < layers[layerNumber].size() - 1; ++n) {
            layers[layerNumber][n].feedForward(previousLayer);
        }
    }
    
}

void Net::backPropergate(const vector<double> &targetValues)
{
    // Calculate overall net error (RMS of output neuron errors) "Root Mean Square Error"
    Layer &outputLayer = layers.back();
    
    error = 0.0;
    
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetValues[n] - outputLayer[n].getOutputValue();
        error += delta * delta;
    }
    
    error /= outputLayer.size() - 1;
    error = sqrt(error);
    
    
    // Implement a recent average measurement
    recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error)
        / (recentAverageSmoothingFactor + 1.0);
    
    // Calculate the output layer gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calculateOutputGradients(targetValues[n]);
    }

    // Calculate gradients on hidden layers
    for (unsigned long layerNumber = layers.size() - 2; layerNumber > 0; --layerNumber) {
        Layer &hiddenLayer = layers[layerNumber];
        Layer &nextLayer = layers[layerNumber + 1];
        
        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calculateHiddenGradidents(nextLayer);
        }
    }
    
    // For all layers from outputs to first hidden layer
    // Update connections weights
    
    for (unsigned long layerNumber = layers.size() - 1; layerNumber > 0; --layerNumber) {
        Layer &layer = layers[layerNumber];
        Layer &previousLayer = layers[layerNumber - 1];
        
        for (unsigned long n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(previousLayer);
        }
    }
}

void Net::getResults(vector<double> &resultsVals) const
{
    resultsVals.clear();
    
    for (unsigned n = 0; n < layers.back().size() - 1; ++n) {
        resultsVals.push_back(layers.back()[n].getOutputValue());
    }
}


int main(int argc, const char * argv[]) {
    
    // e.g. { 3, 2, 1 }
    vector<unsigned> topology;
    
    //Load training data
    TrainingData trainData("data.txt");
    trainData.getTopology(topology);
    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    //Loop until training data is empty
    while (!trainData.isEof()) {
        
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backPropergate(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "
                << myNet.getRecentAverageError() << endl;
    }
    
    return 0;
}
