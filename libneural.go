package main

import (
    "fmt"
    "math"
    "math/rand"
    "github.com/gonum/matrix/mat64"
)

type NeuralNetwork struct {
    NLayers int
    NeuronNum []int
    Layers []*Layer
}

type Layer struct {
    Num int
    Neuron []float64
    Delta []float64
    Weight [][]float64
    Neuronv *mat64.Dense
    BDeltav *mat64.Dense
    SDeltav *mat64.Dense
    Weightv *mat64.Dense
}

func (nn *NeuralNetwork) Init(layers []int) {
    numLayers := len(layers)
    nn.NLayers = numLayers
    nn.Layers = make([]*Layer, 0, len(layers))

    // Init input layer and hidden layer
    for i := 0; i < numLayers-1; i++ {
        a := layers[i]
        b := layers[i+1]
        layer := &Layer{
            Num: a+1,
            Neuron: make([]float64, a),
            Delta: make([]float64, a),
            Weight: InitWeight(a, b),
            Neuronv: InitNeuron(a+1),
            Weightv: InitWeightv(b, a+1),
            BDeltav: InitBigDelta(b+1, a+1),
            SDeltav: InitSmallDelta(a+1),
        }
        if i == numLayers-2 {
            // Big delta in the last hidden layer is differenct
            layer.BDeltav = InitBigDelta(b, a+1)
        } else {
            layer.BDeltav = InitBigDelta(b+1, a+1)
        }
        nn.Layers = append(nn.Layers, layer)
    }

    // Init output layer
    a := layers[numLayers-1]
    layer := &Layer{
        Num: a,
        Neuron: make([]float64, a),
        Delta: make([]float64, a),
        Neuronv: InitNeuron(a),
        SDeltav: InitSmallDelta(a),
    }
    nn.Layers = append(nn.Layers, layer)
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
        //arr[i] = float64(i+1) * 0.01
    }
    m := mat64.NewDense(a, b, arr)
    fmt.Printf("InitWeight\nm = %.3v\n", mat64.Formatted(m, mat64.Prefix("    "), mat64.Squeeze()))
    return m
}

func InitSmallDelta(n int) *mat64.Dense {
    v := mat64.NewDense(n, 1, nil)
    fmt.Printf("InitSmallDelta\nm = %.3v\n", mat64.Formatted(v, mat64.Prefix("    "), mat64.Squeeze()))
    return v
}

func InitBigDelta(a, b int) *mat64.Dense {
    arr := make([]float64, a*b)
    m := mat64.NewDense(a, b, arr)
    fmt.Printf("InitBigDelta\nm = %.3v\n", mat64.Formatted(m, mat64.Prefix("    "), mat64.Squeeze()))
    return m
}

func (nn *NeuralNetwork) Train(patterns [][][]float64, iterations int, lRate, rParam float64) {
	for i := 0; i < iterations; i++ {
        err := 0.0
        cnt := 0
        fmt.Printf("\n\n ====== %d ====== \n\n", i)
		for _, p := range patterns {
            fmt.Printf("p[0]: %v, p[1]: %v\n", p[0], p[1])
			nn.ForwardProp(p[0])
            nn.BackProp(p[1])
            err += nn.CostFunc(p[1])
            cnt++
		}

        err2 := 0.0
        for i := 0; i < nn.NLayers-1; i++ {
            arr := make([]float64, len(nn.Layers[i].Neuronv.RawMatrix().Data))
            copy(arr, nn.Layers[i].Neuronv.RawMatrix().Data)
            for _, v := range arr {
                err2 += v*v
            }
        }
        err = -(1/float64(cnt))*err + (rParam/2*float64(cnt))*err2
        
        nn.GradientDecent(lRate, cnt)
        fmt.Printf("[output] err: %f \n", err)
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
        //fmt.Printf("a' = %.3v\n", mat64.Formatted(nn.Layers[i+1].Neuronv, mat64.Prefix("    "), mat64.Squeeze()))
    }
    fmt.Printf("[output] prediect %.3v -> %.3v\n", patterns, nn.Layers[nn.NLayers-1].Neuronv.RawMatrix().Data)
}

func (nn *NeuralNetwork) BackProp(patterns []float64) {
    // Set output layer sdelta
    arr := make([]float64, len(nn.Layers[nn.NLayers-1].Neuronv.RawMatrix().Data))
    copy(arr, nn.Layers[nn.NLayers-1].Neuronv.RawMatrix().Data)
    for i, _ := range arr {
        arr[i] = arr[i] - patterns[i]
    }
    nn.Layers[nn.NLayers-1].SDeltav.SetCol(0, arr)
    fmt.Printf("d%d = %.3v\n",nn.NLayers-1, mat64.Formatted(nn.Layers[nn.NLayers-1].SDeltav, mat64.Prefix("    "), mat64.Squeeze()))

    for i := nn.NLayers-2; i > 0; i-- {

        sdelta := &mat64.Dense{}
        if i == nn.NLayers-2 {
            sdelta = nn.Layers[i+1].SDeltav
            fmt.Printf("debug = %.3v\n", mat64.Formatted(nn.Layers[i+1].SDeltav, mat64.Prefix("    "), mat64.Squeeze()))
        } else {
            arr := nn.Layers[i+1].SDeltav.RawMatrix().Data
            sdelta = mat64.NewDense(len(arr)-1, 1, arr[1:])
        }
        
        theta := nn.Layers[i].Weightv
        z := &mat64.Dense{}

        fmt.Printf("d%d = %.3v\n", i+1, mat64.Formatted(sdelta, mat64.Prefix("    "), mat64.Squeeze()))
        fmt.Printf("t%d = %.3v\n", i, mat64.Formatted(theta, mat64.Prefix("    "), mat64.Squeeze()))

        z.Mul(theta.T(), sdelta)


        fmt.Printf("z = %.3v\n", mat64.Formatted(z, mat64.Prefix("    "), mat64.Squeeze()))

        gprime := make([]float64, len(nn.Layers[i].Neuronv.RawMatrix().Data))
        copy(gprime, nn.Layers[i].Neuronv.RawMatrix().Data)
        for i, _ := range gprime {
            gprime[i] = gprime[i] * (1-gprime[i])
        }
        fmt.Printf("gprime = %#.3v\n", gprime)

        arr := make([]float64, len(z.RawMatrix().Data))
        copy(arr, z.RawMatrix().Data)
        for i, _ := range gprime {
            arr[i] = arr[i] * gprime[i]
        }

        nn.Layers[i].SDeltav.SetCol(0, arr)
        fmt.Printf("d%d = %.3v\n", i, mat64.Formatted(nn.Layers[i].SDeltav, mat64.Prefix("    "), mat64.Squeeze()))
    }

    for i := 0; i < nn.NLayers-1; i++ {
        z := &mat64.Dense{}
        sdelta := nn.Layers[i+1].SDeltav
        a := nn.Layers[i].Neuronv
        z.Mul(sdelta, a.T())
        fmt.Printf("z = %.3v\n", mat64.Formatted(z, mat64.Prefix("    "), mat64.Squeeze()))
        //fmt.Printf("before bdelta = %.3v\n", mat64.Formatted(nn.Layers[i].BDeltav, mat64.Prefix("    "), mat64.Squeeze()))
        nn.Layers[i].BDeltav.Add(nn.Layers[i].BDeltav, z)
        fmt.Printf("D%d' = %.3v\n", i, mat64.Formatted(nn.Layers[i].BDeltav, mat64.Prefix("    "), mat64.Squeeze()))
    }

}

func (nn *NeuralNetwork) CostFunc(y []float64) float64 {
    a := make([]float64, len(nn.Layers[nn.NLayers-1].Neuronv.RawMatrix().Data))
    copy(a, nn.Layers[nn.NLayers-1].Neuronv.RawMatrix().Data)
    
    if len(y) != len(a) {
        fmt.Printf("y %d not correct \n", len(y))
        return 0.0
    }

    err := 0.0
    for i := 0; i < len(a); i++ {
        err += y[i]*math.Log(a[i]) + (1-y[i])*math.Log(1-a[i])
    }
    return err
}

func (nn *NeuralNetwork) GradientDecent(lRate float64, cnt int)  {
    for i := 0; i < nn.NLayers-1; i++ {
        fmt.Printf("B = %.5v\n", mat64.Formatted(nn.Layers[i].BDeltav, mat64.Prefix("    "), mat64.Squeeze()))

        m, n := nn.Layers[i].BDeltav.Dims()
        z := &mat64.Dense{}
        if i == nn.NLayers-2 {
            z = mat64.DenseCopyOf(nn.Layers[i].BDeltav)
        } else {
            arr := make([]float64, m*n)
            copy(arr, nn.Layers[i].BDeltav.RawMatrix().Data)
            z = mat64.NewDense(m-1, n, arr[n:])
        }

        fmt.Printf("z = %.3v\n", mat64.Formatted(z, mat64.Prefix("    "), mat64.Squeeze()))
        z.Scale(-lRate/float64(cnt), z)
        fmt.Printf("t = %.3v\n", mat64.Formatted(nn.Layers[i].Weightv, mat64.Prefix("    "), mat64.Squeeze()))
        fmt.Printf("z' = %.3v\n", mat64.Formatted(z, mat64.Prefix("    "), mat64.Squeeze()))
        
        //nn.Layers[i].Weightv.Scale(0.99, nn.Layers[i].Weightv)
        //fmt.Printf("t' = %.3v\n", mat64.Formatted(nn.Layers[i].Weightv, mat64.Prefix("    "), mat64.Squeeze()))


        nn.Layers[i].Weightv.Add(nn.Layers[i].Weightv, z)
        fmt.Printf("t'' = %.3v\n", mat64.Formatted(nn.Layers[i].Weightv, mat64.Prefix("    "), mat64.Squeeze()))

        //t := mat64.DenseCopyOf(nn.Layers[i].Weightv)
        //t.Scale(-0.001, t)
        //nn.Layers[i].Weightv.Add(nn.Layers[i].Weightv, t)
        

        nn.Layers[i].BDeltav.Scale(0, nn.Layers[i].BDeltav)
    }
}

func sigmoid_v2(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid_v2(y float64) float64 {
	return y * (1 - y)
}


