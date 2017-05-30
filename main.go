package main

import (
	"fmt"
	"math/rand"
	//"github.com/goml/gobrain"
)
/*
func main() {
	fmt.Printf("\n\n\n\n\n")
	// set the random seed to 0
	rand.Seed(0)

	// create the XOR representation patter to train the network
	
	patterns := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}
	patterns2 := []float64{2, 2,}

	// instantiate the Feed Forward
	ff := FeedForward{}

	// initialize the Neural Network;
	// the networks structure will contain:
	// 2 inputs, 2 hidden nodes and 1 output.
	ff.Init(2, 2, 1)

	// train the network using the XOR patterns
	// the training will run for 1000 epochs
	// the learning rate is set to 0.6 and the momentum factor to 0.4
	// use true in the last parameter to receive reports about the learning error
	ff.Train(patterns, 10000, 0.6, 0.4, true)
	//fmt.Println(a)
	ff.Test(patterns)
	
	ff.Update(patterns2)
	
}
*/

func main() {
	fmt.Printf("\n\n\n\n\n")
	rand.Seed(0)

	nn := NeuralNetwork{}
	nn.Init([]int{2,4,1})

	patterns := [][][]float64{
		{{5, 6}, {0}},
		{{7, 8}, {1}},
		{{9, 10}, {1}},
		{{11, 12}, {0}},
	}

	nn.Train(patterns, 1, 0.6)

}