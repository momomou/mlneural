// Package gobrain provides basic neural networks algorithms.
package main

import (
	"fmt"
	"log"
	"math"
)

// FeedForwad struct is used to represent a simple neural network
type FeedForward struct {
	// Number of input, hidden and output nodes
	NInputs, NHiddens, NOutputs int
	// Whether it is regression or not
	Regression bool
	// Activations for nodes
	InputActivations, HiddenActivations, OutputActivations []float64
	// ElmanRNN contexts
	Contexts [][]float64
	// Weights
	InputWeights, OutputWeights [][]float64
	// Last change in weights for momentum
	InputChanges, OutputChanges [][]float64
}

/*
Initialize the neural network;

the 'inputs' value is the number of inputs the network will have,
the 'hiddens' value is the number of hidden nodes and
the 'outputs' value is the number of the outputs of the network.
*/
func (nn *FeedForward) Init(inputs, hiddens, outputs int) {
	nn.NInputs = inputs + 1   // +1 for bias
	nn.NHiddens = hiddens + 1 // +1 for bias
	nn.NOutputs = outputs

	nn.InputActivations = vector(nn.NInputs, 1.0)
	nn.HiddenActivations = vector(nn.NHiddens, 1.0)
	nn.OutputActivations = vector(nn.NOutputs, 1.0)

	nn.InputWeights = matrix(nn.NInputs, nn.NHiddens)
	nn.OutputWeights = matrix(nn.NHiddens, nn.NOutputs)

	for i := 0; i < nn.NInputs; i++ {
		for j := 0; j < nn.NHiddens; j++ {
			nn.InputWeights[i][j] = random(-1, 1)
		}
	}

	for i := 0; i < nn.NHiddens; i++ {
		for j := 0; j < nn.NOutputs; j++ {
			nn.OutputWeights[i][j] = random(-1, 1)
		}
	}

	nn.InputChanges = matrix(nn.NInputs, nn.NHiddens)
	nn.OutputChanges = matrix(nn.NHiddens, nn.NOutputs)
}

/*
Set the number of contexts to add to the network.

By default the network do not have any context so it is a simple Feed Forward network,
when contexts are added the network behaves like an Elman's SRN (Simple Recurrent Network).

The first parameter (nContexts) is used to indicate the number of contexts to be used,
the second parameter (initValues) can be used to create custom initialized contexts.

If 'initValues' is set, the first parameter 'nContexts' is ignored and
the contexts provided in 'initValues' are used.

When using 'initValues' note that contexts must have the same size of hidden nodes + 1 (bias node).
*/
func (nn *FeedForward) SetContexts(nContexts int, initValues [][]float64) {
	if initValues == nil {
		initValues = make([][]float64, nContexts)

		for i := 0; i < nContexts; i++ {
			initValues[i] = vector(nn.NHiddens, 0.5)
		}
	}

	nn.Contexts = initValues
}

/*
Forward Propagation
The Update method is used to activate the Neural Network.
Given an array of inputs, it returns an array, of length equivalent of number of outputs, with values ranging from 0 to 1.
*/
func (nn *FeedForward) Update(inputs []float64) []float64 {
	if len(inputs) != nn.NInputs-1 {
		log.Fatal("Error: wrong number of inputs")
	}

	for i := 0; i < nn.NInputs-1; i++ {
		//fmt.Printf("nn.InputActivations[%d]: %f -> %f \n", i, nn.InputActivations[i], inputs[i])
		nn.InputActivations[i] = inputs[i]
	}

	for i := 0; i < nn.NHiddens-1; i++ {
		var sum float64

		for j := 0; j < nn.NInputs; j++ {
			sum += nn.InputActivations[j] * nn.InputWeights[j][i]
		}

		// compute contexts sum
		for k := 0; k < len(nn.Contexts); k++ {
			for j := 0; j < nn.NHiddens-1; j++ {
				sum += nn.Contexts[k][j]
			}
		}

		//fmt.Printf("nn.HiddenActivations[%d]: %f -> %f, sum: %f\n", i, nn.HiddenActivations[i], sigmoid(sum), sum)
		nn.HiddenActivations[i] = sigmoid(sum)
	}

	// update the contexts
	if len(nn.Contexts) > 0 {
		for i := len(nn.Contexts) - 1; i > 0; i-- {
			nn.Contexts[i] = nn.Contexts[i-1]
		}
		nn.Contexts[0] = nn.HiddenActivations
	}

	for i := 0; i < nn.NOutputs; i++ {
		var sum float64
		for j := 0; j < nn.NHiddens; j++ {
			sum += nn.HiddenActivations[j] * nn.OutputWeights[j][i]
		}

		//fmt.Printf("nn.OutputActivations[%d]: %f -> %f, sum: %f\n", i, nn.OutputActivations[i], sigmoid(sum), sum)
		nn.OutputActivations[i] = sigmoid(sum)
	}

	return nn.OutputActivations
}
/*
The BackPropagate method is used, when training the Neural Network,
to back propagate the errors from network activation.
*/
func (nn *FeedForward) BackPropagate(targets []float64, lRate, mFactor float64) float64 {
	if len(targets) != nn.NOutputs {
		log.Fatal("Error: wrong number of target values")
	}

	outputDeltas := vector(nn.NOutputs, 0.0)
	for i := 0; i < nn.NOutputs; i++ {
		r := dsigmoid(nn.OutputActivations[i]) * (targets[i] - nn.OutputActivations[i])
		//r2 := nn.OutputActivations[i] - targets[i]
		//fmt.Printf("outputDeltas[%d]: %f -> %f, r2: %f, target: %f\n", i, outputDeltas[i], r, r2, targets[i])
		outputDeltas[i] = r
	}

	hiddenDeltas := vector(nn.NHiddens, 0.0)
	for i := 0; i < nn.NHiddens; i++ {
		var e float64

		for j := 0; j < nn.NOutputs; j++ {
			e += outputDeltas[j] * nn.OutputWeights[i][j]
		}

		r := dsigmoid(nn.HiddenActivations[i]) * e
		//fmt.Printf("hiddenDeltas[%d]: %f -> %f \n", i, hiddenDeltas[i], r)
		hiddenDeltas[i] = r
	}

	for i := 0; i < nn.NHiddens; i++ {
		for j := 0; j < nn.NOutputs; j++ {
			change := outputDeltas[j] * nn.HiddenActivations[i]
			r := nn.OutputWeights[i][j] + lRate*change + mFactor*nn.OutputChanges[i][j]
			//fmt.Printf("OutputWeights[%d][%d]: %f -> %f, change: %f, last change: %f \n",
			//				i, j, nn.OutputWeights[i][j], r, change, nn.OutputChanges[i][j])
			nn.OutputWeights[i][j] = r
			nn.OutputChanges[i][j] = change
		}
	}

	for i := 0; i < nn.NInputs; i++ {
		for j := 0; j < nn.NHiddens; j++ {
			change := hiddenDeltas[j] * nn.InputActivations[i]
			r := nn.InputWeights[i][j] + lRate*change + mFactor*nn.InputChanges[i][j]
			//fmt.Printf("InputWeights[%d][%d]: %f -> %f, change: %f, last change: %f \n",
			//				i, j, nn.InputWeights[i][j], r, change, nn.InputChanges[i][j])
			nn.InputWeights[i][j] = r
			nn.InputChanges[i][j] = change
		}
	}

	var e float64

	for i := 0; i < len(targets); i++ {
		e += 0.5 * math.Pow(targets[i]-nn.OutputActivations[i], 2)
	}

	return e
}
/*
This method is used to train the Network, it will run the training operation for 'iterations' times
and return the computed errors when training.
*/
func (nn *FeedForward) Train(patterns [][][]float64, iterations int, lRate, mFactor float64, debug bool) []float64 {
	errors := make([]float64, iterations)

	for i := 0; i < iterations; i++ {
		var e float64
		//fmt.Printf("Iteration #%d \n", i)
		for _, p := range patterns {
			nn.Update(p[0])

			tmp := nn.BackPropagate(p[1], lRate, mFactor)
			e += tmp
		}

		errors[i] = e

		if debug && i%10 == 0 {
			fmt.Println(i, e)
		}
	}

	return errors
}

func (nn *FeedForward) Test(patterns [][][]float64) {
	for _, p := range patterns {
		fmt.Println(p[0], "->", nn.Update(p[0]), " : ", p[1])
	}
}
