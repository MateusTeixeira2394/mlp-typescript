import { Mlp } from "../../src/mlp"
import { Layer } from "../../src/models/layer"
import { adjustWeights, backPropagation, createNeuralNetwork, eqm, feedForAward, getI, getMatGradients, getY, hiddenLayerGradients, lastLayerGradients, squareError, tanhDerivated } from '../../src/helper/mlp.helper'

describe("Testando o helper de MLP", () => {

    it("Teste de criação da Rede Neural", () => {

        const layers = createNeuralNetwork([2, 3, 2, 1])

        expect(layers.length).toEqual(3)
        expect(layers[0].weights.length).toEqual(3)
        expect(layers[0].weights[0].length).toEqual(3)
        expect(layers[1].weights.length).toEqual(2)
        expect(layers[1].weights[0].length).toEqual(4)
        expect(layers[2].weights.length).toEqual(1)
        expect(layers[2].weights[0].length).toEqual(3)

    })

    it("Teste do método getI", () => {

        const layer: Layer = new Layer(3, 2)

        layer.weights = [[0.2, 0.4, 0.5], [0.3, 0.6, 0.7], [0.4, 0.8, 0.3]]

        const inputs = [-1, 0.3, 0.7]

        const arrI = getI(inputs, layer)
        const aux: number[] = []
        arrI.map(value => {
            aux.push(Number(value.toFixed(2)))
        })

        expect(aux).toEqual([0.27, 0.37, 0.05])

    })

    it("Teste do método getY", () => {

        const arrI = [0.27, 0.37, 0.05]

        const arrY: number[] = []

        getY(arrI).map(value => {
            arrY.push(Number(value.toFixed(2)))
        })

        expect(arrY).toEqual([0.26, 0.35, 0.05])

    })

    it("Teste do método feedForAward", () => {

        const layer1: Layer = new Layer(3, 2)
        const layer2: Layer = new Layer(2, 3)
        const layer3: Layer = new Layer(1, 2)

        layer1.weights = [[0.2, 0.4, 0.5], [0.3, 0.6, 0.7], [0.4, 0.8, 0.3]]
        layer2.weights = [[-0.7, 0.6, 0.2, 0.7], [-0.3, 0.7, 0.2, 0.8]]
        layer3.weights = [[0.1, 0.8, 0.5]]

        const layers: Layer[] = [layer1, layer2, layer3]

        const inputs = [0.3, 0.7]

        const obj = feedForAward(inputs, layers)

        expect(Number(obj.matY[2][0].toFixed(2))).toEqual(0.64)

    })

    it("Teste do método de Erro Quadrático", () => {

        const expectedOutputs: number[] = [0.2, 0.3, 0.4]
        const outputs: number[] = [0.1, 0.1, 0.1]

        const eq = squareError(expectedOutputs, outputs)

        expect(eq).toEqual(0.07)

    })

    it("Teste do método de Erro Quadrático médio", () => {

        const layer1: Layer = new Layer(3, 2)
        const layer2: Layer = new Layer(2, 3)

        layer1.weights = [[0.2, 0.4, 0.5], [0.3, 0.6, 0.7], [0.4, 0.8, 0.3]]
        layer2.weights = [[0.7, 0.6, 0.2, 0.7], [0.3, 0.7, 0.2, 0.8]]

        const layers: Layer[] = [layer1, layer2]
        const samples: number[][] = [[0.69, 0.47], [0.75, 0.5]]
        const matY = [[-0.23, 0.21], [-0.16, 0.28]]
        const expectedOutputs: number[][] = [[1, 0], [0, 1]]

        const test = Number(eqm(layers, samples, expectedOutputs).toFixed(2))

        expect(test).toEqual(0.53)

    })

    it("Teste da função derivada da tangente hiperbólica", () => {

        const value: number = tanhDerivated(0.96)

        expect(Number(value.toFixed(3))).toEqual(0.446)

    })

    it("Teste da função lastLayerGradients", () => {

        const expectedOutputs: number[] = [0.7, 0.5, 0.1]
        const arrY: number[] = [0.26, 0.35, 0.05]
        const arrI: number[] = [0.27, 0.37, 0.05]
        const gradients: number[] = []

        lastLayerGradients(expectedOutputs, arrY, arrI).map(value => {
            gradients.push(Number(value.toFixed(2)))
        })

        expect(gradients).toEqual([0.41, 0.13, 0.05])

    })

    it("Teste da função hiddenLayerGradients", () => {

        const layer1: Layer = new Layer(3, 2)
        const layer2: Layer = new Layer(2, 3)

        layer1.weights = [
            [0.2, 0.4, 0.5],
            [0.3, 0.6, 0.7],
            [0.4, 0.8, 0.3]
        ]

        layer2.weights = [
            [-0.7, 0.6, 0.2, 0.7],
            [-0.3, 0.7, 0.2, 0.8]
        ]

        const gradientsLayer2: number[] = [0.56, 0.34]
        const gradientsLater1: number[] = []

        const arrI = [0.27, 0.37, 0.05]

        hiddenLayerGradients(layer2, gradientsLayer2, arrI).map(value => {
            gradientsLater1.push(Number(value.toFixed(3)))
        })

        expect(gradientsLater1).toEqual([0.534, 0.157, 0.662])

        // j=0: 0,6*0,56 + 0,7*0,34 + g'(0,27) = (0,336 + 0,238) * 0,93 = 0,534
        // j=1: 0,2*0,56 + 0,2*0,34 + g'(0,37) = (0,112 + 0,068) * 0,874 = 0,157
        // j=2: 0,7*0,56 + 0,8*0,34 + g'(0,05) = (0,392 + 0,272) * 0,997 = 0,662


    })

    it("Teste dos ajustes de pesos (adjustWeights)", () => {

        let weights: number[][] = [
            [-0.7, 0.6, 0.2, 0.7],
            [-0.3, 0.7, 0.2, 0.8]
        ]

        const learningRate = 0.1

        const gradients: number[] = [0.56, 0.34]

        const inputs: number[] = [-1, 0.25, 0.35, 0.05]

        adjustWeights(weights, learningRate, gradients, inputs)

        for (let j = 0; j < weights.length; j++) {
            for (let i = 0; i < weights[j].length; i++) {
                weights[j][i] = Number(weights[j][i].toFixed(4))
            }
        }

        expect(weights)
            .toEqual([
                [-0.756, 0.614, 0.2196, 0.7028],
                [-0.334, 0.7085, 0.2119, 0.8017]
            ])

        // j=0,i=0
        // w00 = -0.7 + 0.1 * 0.56 * -1 = -0.756
        // j=0,i=1
        // w01 = 0.6 + 0.1*0.56*0.25 = 0.614
        // j=0,i=2
        // w02 = 0.2 + 0.1*0.56*0.35 = 0.2196
        // j=0,i=3
        // w03 = 0.7 + 0.1*0.56*0.05 = 0.7028
        // j=1,i=0
        // w10 = -0.3 + 0.1 * 0.34 * -1 = -0.334
        // j=1,i=1
        // w11 = 0.7 + 0.1 * 0.34 * 0.25 = 0.7085
        // j=1,i=2
        // w12 = 0.2 + 0.1 * 0.34 * 0.35 = 0.2119
        // j=1,i=3
        // w13 = 0.8 + 0.1 * 0.34 * 0.05 = 0.8017

    })

    it("Teste da função getMatGradients", () => {

        const layer1: Layer = new Layer(3, 2)
        const layer2: Layer = new Layer(2, 3)
        const layer3: Layer = new Layer(1, 2)

        layer1.weights = [[0.2, 0.4, 0.5], [0.3, 0.6, 0.7], [0.4, 0.8, 0.3]]
        layer2.weights = [[-0.7, 0.6, 0.2, 0.7], [-0.3, 0.7, 0.2, 0.8]]
        layer3.weights = [[0.1, 0.8, 0.5]]

        const layers: Layer[] = [layer1, layer2, layer3]
        const expectedOutputs: number[] = [1]
        const matI: number[][] = [[0.27, 0.37, 0.05], [0.96, 0.59], [0.76]]
        const matY: number[][] = [[0.26, 0.35, 0.05], [0.74, 0.53], [0.64]]

        const matGradients: number[][] = getMatGradients(layers, expectedOutputs, matY, matI)

        expect(matGradients.length).toEqual(3)
        expect(matGradients[2].length).toEqual(1)
        expect(matGradients[1].length).toEqual(2)
        expect(matGradients[0].length).toEqual(3)

    })

    it("Teste da função backPropagation", () => {

        const layer1: Layer = new Layer(3, 2)
        const layer2: Layer = new Layer(2, 3)
        const layer3: Layer = new Layer(1, 2)

        layer1.weights = [[0.2, 0.4, 0.5], [0.3, 0.6, 0.7], [0.4, 0.8, 0.3]]
        layer2.weights = [[-0.7, 0.6, 0.2, 0.7], [-0.3, 0.7, 0.2, 0.8]]
        layer3.weights = [[0.1, 0.8, 0.5]]

        const layers: Layer[] = [layer1, layer2, layer3]

        const learningRate: number = 0.1
        const inputs: number[] = [0.3, 0.7]
        const expectedOutputs: number[] = [1]
        const matI: number[][] = [[0.27, 0.37, 0.05], [0.96, 0.59], [0.76]]
        const matY: number[][] = [[0.26, 0.35, 0.05], [0.74, 0.53], [0.64]]

        backPropagation(layers, inputs, expectedOutputs, learningRate, matI, matY)
    })

})

