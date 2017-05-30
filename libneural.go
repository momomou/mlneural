package main

import (
    "fmt"
    "math"
    "math/rand"
    "github.com/gonum/matrix/mat64"
)

type NeuralNetwork struct {
    NLayers int
    NEachLayer []int
    Layers []*Layer
}

type Layer struct {
    Num int
    Neuron []float64
    Delta []float64
    Weight [][]float64
    Neuronv *mat64.Dense
    Deltav *mat64.Dense
    Weightv *mat64.Dense
}

func (nn *NeuralNetwork) Init(layers []int) {
    numLayers := len(layers)
    fmt.Printf("nn.Layers: %d %d \n", len(nn.Layers), cap(nn.Layers))
    nn.NLayers = numLayers
    nn.Layers = make([]*Layer, 0, len(layers))
    fmt.Printf("nn.Layers: %d %d \n", len(nn.Layers), cap(nn.Layers))
    
    // Init input layer and hidden layer
    for i := 0; i < numLayers-1; i++ {
        a := layers[i]
        b := layers[i+1]
        layer := &Layer{
            Num: a,
            Neuron: make([]float64, a),
            Delta: make([]float64, a),
            Weight: InitWeight(a, b),
            Neuronv: InitNeuron(a+1),         // +1 for bias
            Weightv: InitWeightv(b, a+1),       // +1 for bias
        }
        nn.Layers = append(nn.Layers, layer)
        fmt.Printf("nn.Layers: %d %d \n", len(nn.Layers), cap(nn.Layers))
        //fmt.Printf("nn.Layers: %#v \n", nn.Layers[i])
    }

    // Init output layer
    a := layers[numLayers-1]
    layer := &Layer{
        Num: a,
        Neuron: make([]float64, a),
        Delta: make([]float64, a),
        Neuronv: InitNeuron(a),
    }
    nn.Layers = append(nn.Layers, layer)
    fmt.Printf("nn.Layers: %d %d \n", len(nn.Layers), cap(nn.Layers))
    fmt.Printf("nn.Layers: %#v \n", nn.Layers[numLayers-1])
}

func InitWeight(a, b int) [][]float64 {
	arr := make([][]float64, a)
	for i, _ := range arr {
		arr[i] = make([]float64, b)
	}
    for i, _ := range arr {
        for j, _ := range arr[i] {
            arr[i][j] = 2*rand.Float64() - 1
        }
	}
    return arr
}

func InitNeuron(n int) *mat64.Dense {
    v := mat64.NewDense(n, 1, nil)
    v.Set(0, 0, 1)
    fmt.Printf("InitNeuron\nm = %.3v\n", mat64.Formatted(v, mat64.Prefix("    "), mat64.Squeeze()))
    return v
}

func InitWeightv(a, b int) *mat64.Dense {
    arr := make([]float64, a*b)
    for i, _ := range arr {
        arr[i] = 2*rand.Float64() - 1
    }
    m := mat64.NewDense(a, b, arr)
    fmt.Printf("InitWeight\nm = %.3v\n", mat64.Formatted(m, mat64.Prefix("    "), mat64.Squeeze()))
    return m
}

func (nn *NeuralNetwork) Train(patterns [][][]float64, iterations int, lRate float64) {
	for i := 0; i < iterations; i++ {
		for _, p := range patterns {
			nn.ForwardProp(p[0])
            /*
            fmt.Printf("p: %#v \n", p)
            fmt.Printf("p[0]: %#v \n", p[0])
            fmt.Printf("p[1]: %#v \n", p[1])

            a := mat64.NewDense(2, 2, []float64{2,3,4,5})
            b := mat64.NewDense(2, 1, []float64{6,7})

            fmt.Printf("a = %v\n", mat64.Formatted(a, mat64.Prefix("    "), mat64.Squeeze()))
            fmt.Printf("b = %v\n", mat64.Formatted(b, mat64.Prefix("    "), mat64.Squeeze()))

            c := &mat64.Dense{}
            c.Mul(a, b)
            fmt.Printf("c = %v\n", mat64.Formatted(c, mat64.Prefix("    "), mat64.Squeeze()))

            a1, a2 := c.Dims()
            fmt.Printf("%d %d\n", a1, a2)
            */

		}
	}
}

func (nn *NeuralNetwork) ForwardProp(patterns []float64) {

    // Set input to input layer
    nn.Layers[0].Neuronv.SetCol(0, append([]float64{1}, patterns...))

    for i := 0; i < nn.NLayers-1; i++ {
        z := &mat64.Dense{}
        a := nn.Layers[i].Neuronv
        theta := nn.Layers[i].Weightv
        z.Mul(theta, a)
        arr := make([]float64, len(z.RawMatrix().Data))
        copy(arr, z.RawMatrix().Data)
        for j, _ := range arr {
            arr[j] = sigmoid_v2(arr[j])
        }

        if i < nn.NLayers-2 {
            // Set bias to the first neuron
            nn.Layers[i+1].Neuronv.SetCol(0, append([]float64{1}, arr...))
        } else {
            // No need bias for output layer
            nn.Layers[i+1].Neuronv.SetCol(0, arr)
        }
        fmt.Printf("a%d = %.3v\n", i, mat64.Formatted(a, mat64.Prefix("    "), mat64.Squeeze()))
        fmt.Printf("t = %.3v\n", mat64.Formatted(theta, mat64.Prefix("    "), mat64.Squeeze()))
        fmt.Printf("z = %.3v\n", mat64.Formatted(z, mat64.Prefix("    "), mat64.Squeeze()))
        fmt.Printf("a' = %.3v\n", mat64.Formatted(nn.Layers[i+1].Neuronv, mat64.Prefix("    "), mat64.Squeeze()))
    }
}

func sigmoid_v2(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid_v2(y float64) float64 {
	return y * (1 - y)
}
