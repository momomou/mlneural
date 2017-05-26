package main

import (
	"fmt"
	"math/rand"
	//"github.com/goml/gobrain"
)

func main() {
	
	// set the random seed to 0
	rand.Seed(0)

	// create the XOR representation patter to train the network
	
	patterns := [][][]float64{
		{{0, 0}, {0, 0}},
		{{0, 1}, {1, 1}},
		{{1, 0}, {1, 1}},
		{{1, 1}, {0, 0}},
	}
	patterns2 := []float64{2, 2,}

	// instantiate the Feed Forward
	ff := FeedForward{}

	// initialize the Neural Network;
	// the networks structure will contain:
	// 2 inputs, 2 hidden nodes and 1 output.
	ff.Init(2, 3, 2)

	// train the network using the XOR patterns
	// the training will run for 1000 epochs
	// the learning rate is set to 0.6 and the momentum factor to 0.4
	// use true in the last parameter to receive reports about the learning error
	a := ff.Train(patterns, 10, 0.6, 0.4, true)
	fmt.Println(a)
	ff.Test(patterns)
	
	ff.Update(patterns2)
	
}